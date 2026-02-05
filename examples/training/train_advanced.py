#!/usr/bin/env python3
"""
Training with AdvancedPBR reward scheme.

This script uses an advanced reward scheme that combines:
1. Position-Based Returns (PBR) - rewards being in the right position
2. Trading Penalty - discourages overtrading to reduce commission costs
3. Hold Bonus - rewards patience in flat/uncertain markets

The goal is to generate actual profits, not just minimize losses.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.env.default.actions import BSH
from tensortrade.env.default.rewards import AdvancedPBR
import tensortrade.env.default as default


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add scale-invariant features."""
    df = df.copy()

    # Returns
    for p in [1, 4, 12, 24, 48]:
        df[f'ret_{p}h'] = np.tanh(df['close'].pct_change(p) * 10)

    # RSI normalized
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = (100 - (100 / (1 + rs)) - 50) / 50

    # Trend
    sma20 = df['close'].rolling(20).mean()
    sma50 = df['close'].rolling(50).mean()
    df['trend_20'] = np.tanh((df['close'] - sma20) / sma20 * 10)
    df['trend_50'] = np.tanh((df['close'] - sma50) / sma50 * 10)
    df['trend_strength'] = np.tanh((sma20 - sma50) / sma50 * 20)

    # Volatility
    df['vol'] = df['close'].rolling(24).std() / df['close']
    df['vol_norm'] = np.tanh((df['vol'] - df['vol'].rolling(72).mean()) / df['vol'].rolling(72).std())

    # Volume
    df['vol_ratio'] = np.log1p(df['volume'] / df['volume'].rolling(20).mean())

    # BB position
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_pos'] = ((df['close'] - (bb_mid - 2*bb_std)) / (4*bb_std)).clip(0, 1)

    return df.bfill().ffill()


class Callbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        env = base_env.get_sub_environments()[env_index]
        if hasattr(env, 'portfolio'):
            episode.user_data["initial"] = float(env.portfolio.net_worth)

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        env = base_env.get_sub_environments()[env_index]
        if hasattr(env, 'portfolio'):
            final = float(env.portfolio.net_worth)
            initial = episode.user_data.get("initial", 10000)
            episode.custom_metrics["pnl"] = final - initial
            episode.custom_metrics["pnl_pct"] = (final - initial) / initial * 100

            # Track reward scheme stats if available
            if hasattr(env, 'reward_scheme') and hasattr(env.reward_scheme, 'get_stats'):
                stats = env.reward_scheme.get_stats()
                episode.custom_metrics["trade_count"] = stats.get("trade_count", 0)
                episode.custom_metrics["hold_count"] = stats.get("hold_count", 0)


def create_env(config: Dict[str, Any]):
    data = pd.read_csv(config["csv_filename"], parse_dates=['date']).bfill().ffill()

    price = Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
    commission = config.get("commission", 0.0005)
    exchange = Exchange("exchange", service=execute_order,
                       options=ExchangeOptions(commission=commission))(price)

    cash = Wallet(exchange, config.get("initial_cash", 10000) * USD)
    asset = Wallet(exchange, 0 * BTC)
    portfolio = Portfolio(USD, [cash, asset])

    features = [Stream.source(list(data[c]), dtype="float").rename(c)
                for c in config.get("feature_cols", [])]
    feed = DataFeed(features)
    feed.compile()

    # Use AdvancedPBR reward scheme with configurable parameters
    reward_scheme = AdvancedPBR(
        price=price,
        pbr_weight=config.get("pbr_weight", 1.0),
        trade_penalty=config.get("trade_penalty", -0.001),
        hold_bonus=config.get("hold_bonus", 0.0001),
        volatility_threshold=config.get("volatility_threshold", 0.001)
    )
    action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme)

    env = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        window_size=config.get("window_size", 10),
        max_allowed_loss=config.get("max_allowed_loss", 0.4)
    )
    env.portfolio = portfolio
    env.reward_scheme = reward_scheme
    return env


def evaluate(algo, data: pd.DataFrame, feature_cols: list, config: Dict, n: int = 10) -> tuple:
    """Evaluate and return average P&L and trade count."""
    csv = '/tmp/eval_advanced.csv'
    data.reset_index(drop=True).to_csv(csv, index=False)

    env_config = {
        "csv_filename": csv,
        "feature_cols": feature_cols,
        "window_size": config["window_size"],
        "max_allowed_loss": config["max_allowed_loss"],
        "commission": config["commission"],
        "pbr_weight": config.get("pbr_weight", 1.0),
        "trade_penalty": config.get("trade_penalty", -0.001),
        "hold_bonus": config.get("hold_bonus", 0.0001),
        "volatility_threshold": config.get("volatility_threshold", 0.001),
        "initial_cash": 10000,
    }

    pnls = []
    trade_counts = []
    for _ in range(n):
        env = create_env(env_config)
        obs, _ = env.reset()
        done = truncated = False
        while not done and not truncated:
            action = algo.compute_single_action(obs)
            obs, _, done, truncated, _ = env.step(action)
        pnls.append(env.portfolio.net_worth - 10000)
        if hasattr(env.reward_scheme, 'get_stats'):
            trade_counts.append(env.reward_scheme.get_stats()["trade_count"])

    os.remove(csv)
    avg_trades = np.mean(trade_counts) if trade_counts else 0
    return np.mean(pnls), avg_trades


def main():
    print("=" * 70)
    print("TensorTrade - Advanced PBR Training")
    print("=" * 70)
    print("\nReward Components:")
    print("  1. PBR: Position-based returns")
    print("  2. Trade Penalty: Discourages overtrading")
    print("  3. Hold Bonus: Rewards patience in flat markets")

    # Load and prepare data
    print("\nLoading data...")
    cdd = CryptoDataDownload()
    data = cdd.fetch("Bitfinex", "USD", "BTC", "1h")
    data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values('date', inplace=True)
    data.reset_index(drop=True, inplace=True)

    print("Adding features...")
    data = add_features(data)
    feature_cols = [c for c in data.columns if c not in ['date', 'open', 'high', 'low', 'close', 'volume']]
    print(f"Features: {len(feature_cols)}")

    # Split data
    test_candles = 30 * 24
    val_candles = 30 * 24

    test_data = data.iloc[-test_candles:].copy()
    val_data = data.iloc[-(test_candles + val_candles):-test_candles].copy()
    train_data = data.iloc[:-(test_candles + val_candles)].tail(3000).reset_index(drop=True)

    # Buy-and-hold baselines
    val_bh = 10000 * (val_data['close'].iloc[-1] - val_data['close'].iloc[0]) / val_data['close'].iloc[0]
    test_bh = 10000 * (test_data['close'].iloc[-1] - test_data['close'].iloc[0]) / test_data['close'].iloc[0]

    print(f"\nData splits:")
    print(f"  Train: {len(train_data)} candles")
    print(f"  Val:   {len(val_data)} candles (B&H: ${val_bh:+,.0f})")
    print(f"  Test:  {len(test_data)} candles (B&H: ${test_bh:+,.0f})")

    # Save train data
    train_csv = '/tmp/train_advanced.csv'
    train_data.to_csv(train_csv, index=False)

    # Advanced PBR hyperparameters - balanced for active trading
    reward_config = {
        "pbr_weight": 1.0,           # Full PBR weight - reward good trades
        "trade_penalty": -0.0005,    # Small penalty (roughly matches commission)
        "hold_bonus": 0.0,           # No hold bonus - focus on trading
        "volatility_threshold": 0.001,
    }

    env_config = {
        "csv_filename": train_csv,
        "feature_cols": feature_cols,
        "window_size": 17,
        "max_allowed_loss": 0.32,
        "commission": 0.00013,
        "initial_cash": 10000,
        **reward_config,
    }

    # Initialize Ray
    ray.init(num_cpus=6, ignore_reinit_error=True, log_to_driver=False)
    register_env("TradingEnv", create_env)

    # PPO config - using best hyperparameters from Optuna
    config = (
        PPOConfig().api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="TradingEnv", env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=4)
        .callbacks(Callbacks)
        .training(
            lr=3.29e-05,
            gamma=0.992,
            lambda_=0.9,
            clip_param=0.123,
            entropy_coeff=0.015,
            train_batch_size=2000,
            minibatch_size=256,
            num_epochs=7,
            vf_clip_param=100.0,
            model={"fcnet_hiddens": [128, 128], "fcnet_activation": "tanh"},
        )
        .resources(num_gpus=0)
    )

    algo = config.build()

    # Training loop
    print(f"\n{'='*70}")
    print("Training with AdvancedPBR Reward Scheme")
    print(f"{'='*70}")
    print(f"\nReward Config:")
    print(f"  PBR Weight: {reward_config['pbr_weight']}")
    print(f"  Trade Penalty: {reward_config['trade_penalty']}")
    print(f"  Hold Bonus: {reward_config['hold_bonus']}")
    print(f"  Volatility Threshold: {reward_config['volatility_threshold']}")

    train_iters = 80
    best_val_pnl = float('-inf')
    best_iter = 0

    print(f"\nTraining for {train_iters} iterations...")

    for i in range(train_iters):
        result = algo.train()

        if (i + 1) % 10 == 0:
            pnl = result.get('env_runners', {}).get('custom_metrics', {}).get('pnl_mean', 0)
            trades = result.get('env_runners', {}).get('custom_metrics', {}).get('trade_count_mean', 0)

            # Validation
            val_config = {**env_config, "window_size": 17, "max_allowed_loss": 0.32, "commission": 0.00013}
            val_pnl, val_trades = evaluate(algo, val_data, feature_cols, val_config, n=5)

            status = ""
            if val_pnl > best_val_pnl:
                best_val_pnl = val_pnl
                best_iter = i + 1
                status = " *BEST*"

            print(f"  Iter {i+1:3d}: Train P&L ${pnl:+,.0f} ({trades:.0f} trades) | "
                  f"Val P&L ${val_pnl:+,.0f} ({val_trades:.0f} trades){status}")

    # Final evaluation
    print(f"\n{'='*70}")
    print("Final Test Results (Held-Out Data)")
    print(f"{'='*70}")

    test_config = {**env_config, "window_size": 17, "max_allowed_loss": 0.32, "commission": 0.00013}
    test_pnl, test_trades = evaluate(algo, test_data, feature_cols, test_config, n=30)

    print(f"\nTest period: {test_data['date'].iloc[0].date()} to {test_data['date'].iloc[-1].date()}")
    print(f"BTC: ${test_data['close'].iloc[0]:,.0f} -> ${test_data['close'].iloc[-1]:,.0f}")

    print(f"\nAgent (AdvancedPBR):  ${test_pnl:+,.0f} ({test_trades:.0f} avg trades)")
    print(f"Buy & Hold:           ${test_bh:+,.0f}")

    diff = test_pnl - test_bh
    if diff > 0:
        print(f"\n*** AGENT WINS by ${diff:+,.0f}! ***")
    elif test_pnl > 0:
        print(f"\nAgent profitable! (+${test_pnl:,.0f}) but B&H better by ${-diff:,.0f}")
    else:
        print(f"\nB&H wins by ${-diff:+,.0f}")

    # Cleanup
    os.remove(train_csv)
    algo.stop()
    ray.shutdown()

    print("\nDone!")


if __name__ == "__main__":
    main()
