#!/usr/bin/env python3
"""
Simple Trend-Following Strategy Training.

The key insight: complex models overfit. Simple trend-following rules
have been profitable for decades. Let's train a minimal agent that
learns basic trend signals.

Features: Only 5 trend indicators
Network: Tiny (32x32)
Goal: Learn when to be IN market (holding BTC) vs OUT (holding USD)
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
from tensortrade.env.default.rewards import PBR
import tensortrade.env.default as default


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add only 5 simple trend features - minimal to prevent overfitting."""
    df = df.copy()

    # 1. Trend direction: Are we above or below 50-period SMA? (-1 to 1)
    sma50 = df['close'].rolling(50).mean()
    df['trend'] = np.tanh((df['close'] - sma50) / sma50 * 10)

    # 2. Momentum: 24-hour return, clipped
    df['momentum'] = np.tanh(df['close'].pct_change(24) * 10)

    # 3. RSI normalized to -1 to 1
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    df['rsi_norm'] = (rsi - 50) / 50

    # 4. Volatility regime: High or low vol? (0 to 1)
    vol = df['close'].rolling(24).std() / df['close']
    vol_ma = vol.rolling(72).mean()
    df['vol_regime'] = np.tanh((vol - vol_ma) / vol_ma * 5)

    # 5. Recent trend strength: SMA10 vs SMA30
    sma10 = df['close'].rolling(10).mean()
    sma30 = df['close'].rolling(30).mean()
    df['trend_strength'] = np.tanh((sma10 - sma30) / sma30 * 20)

    df = df.bfill().ffill()
    return df


class SimpleCallbacks(DefaultCallbacks):
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


def create_env(config: Dict[str, Any]):
    data = pd.read_csv(config["csv_filename"], parse_dates=['date']).bfill().ffill()

    price = Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
    exchange = Exchange("exchange", service=execute_order,
                       options=ExchangeOptions(commission=0.0005))(price)  # Lower commission

    cash = Wallet(exchange, config.get("initial_cash", 10000) * USD)
    asset = Wallet(exchange, 0 * BTC)
    portfolio = Portfolio(USD, [cash, asset])

    features = [Stream.source(list(data[c]), dtype="float").rename(c)
                for c in config.get("feature_cols", [])]
    feed = DataFeed(features)
    feed.compile()

    reward_scheme = PBR(price=price)
    action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme)

    env = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        window_size=config.get("window_size", 10),
        max_allowed_loss=config.get("max_allowed_loss", 0.3)
    )
    env.portfolio = portfolio
    return env


def evaluate(algo, data, feature_cols, n=20):
    csv = '/tmp/eval.csv'
    data.reset_index(drop=True).to_csv(csv, index=False)
    cfg = {"csv_filename": csv, "feature_cols": feature_cols, "window_size": 10,
           "max_allowed_loss": 0.3, "initial_cash": 10000}

    results = []
    for _ in range(n):
        env = create_env(cfg)
        obs, _ = env.reset()
        done = truncated = False
        while not done and not truncated:
            action = algo.compute_single_action(obs)
            obs, _, done, truncated, _ = env.step(action)
        results.append(env.portfolio.net_worth - 10000)

    os.remove(csv)
    return np.mean(results), np.std(results)


def main():
    print("=" * 70)
    print("TensorTrade - Simple Trend-Following Strategy")
    print("=" * 70)

    # Get data
    print("\nFetching data...")
    cdd = CryptoDataDownload()
    data = cdd.fetch("Bitfinex", "USD", "BTC", "1h")
    data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values('date', inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Add minimal trend features
    print("Adding 5 trend features...")
    data = add_trend_features(data)
    feature_cols = ['trend', 'momentum', 'rsi_norm', 'vol_regime', 'trend_strength']

    # Split: older for train, recent for test
    test_candles = 30 * 24
    val_candles = 30 * 24

    test_data = data.iloc[-test_candles:].copy()
    val_data = data.iloc[-(test_candles + val_candles):-test_candles].copy()
    train_data = data.iloc[:-(test_candles + val_candles)].tail(3000).reset_index(drop=True)

    print(f"\nTrain: {len(train_data)} candles | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"Train period: {train_data['date'].iloc[0].date()} to {train_data['date'].iloc[-1].date()}")
    print(f"Test period: {test_data['date'].iloc[0].date()} to {test_data['date'].iloc[-1].date()}")

    # Buy and hold baselines
    val_bh = 10000 * (val_data['close'].iloc[-1] - val_data['close'].iloc[0]) / val_data['close'].iloc[0]
    test_bh = 10000 * (test_data['close'].iloc[-1] - test_data['close'].iloc[0]) / test_data['close'].iloc[0]
    print(f"\nBuy-and-Hold baselines: Val ${val_bh:+,.0f} | Test ${test_bh:+,.0f}")

    # Save train data
    train_csv = os.path.join(os.getcwd(), 'train_trend.csv')
    train_data.to_csv(train_csv, index=False)

    ray.init(num_cpus=6, ignore_reinit_error=True, log_to_driver=False)
    register_env("TradingEnv", create_env)

    cfg = {"csv_filename": train_csv, "feature_cols": feature_cols,
           "window_size": 10, "max_allowed_loss": 0.3, "initial_cash": 10000}

    # Tiny network, high entropy
    config = (
        PPOConfig().api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="TradingEnv", env_config=cfg)
        .framework("torch")
        .env_runners(num_env_runners=4)
        .callbacks(SimpleCallbacks)
        .training(
            lr=5e-5,
            gamma=0.99,
            lambda_=0.9,
            clip_param=0.1,
            entropy_coeff=0.1,  # Very high entropy
            train_batch_size=4000,
            minibatch_size=128,
            num_epochs=5,
            vf_clip_param=100.0,
            model={"fcnet_hiddens": [32, 32], "fcnet_activation": "tanh"},
        )
        .resources(num_gpus=0)
    )

    algo = config.build()

    print(f"\n{'='*70}")
    print("Training (tiny model, 5 features, high exploration)")
    print(f"{'='*70}")

    import time
    start = time.time()
    best_val = float('-inf')
    best_iter = 0
    patience = 0
    max_patience = 3  # Stop after 3 validation checks without improvement

    for i in range(100):
        result = algo.train()
        reward = result.get('env_runners', {}).get('episode_return_mean', 0)
        pnl = result.get('env_runners', {}).get('custom_metrics', {}).get('pnl_mean', 0)

        # Validate every 10 iterations
        if (i + 1) % 10 == 0:
            val_pnl, val_std = evaluate(algo, val_data, feature_cols, n=10)
            vs_bh = val_pnl - val_bh

            if val_pnl > best_val:
                best_val = val_pnl
                best_iter = i + 1
                patience = 0
                status = "*BEST*"
            else:
                patience += 1
                status = f"wait {patience}/{max_patience}"

            print(f"Iter {i+1:3d} | Train: {reward:+8,.0f} P&L ${pnl:+6,.0f} | "
                  f"Val: ${val_pnl:+6,.0f}±{val_std:.0f} vs B&H ${vs_bh:+6,.0f} | {status}")

            if patience >= max_patience:
                print(f"\nEarly stop at {i+1} (best was {best_iter})")
                break
        elif (i + 1) % 5 == 0:
            print(f"Iter {i+1:3d} | Train: {reward:+8,.0f} P&L ${pnl:+6,.0f}")

    print(f"\nTraining: {time.time()-start:.0f}s, best val at iter {best_iter}")

    # Final test
    print(f"\n{'='*70}")
    print("Final Test (30 episodes)")
    print(f"{'='*70}")

    test_pnl, test_std = evaluate(algo, test_data, feature_cols, n=30)

    print(f"\nTest period: {test_data['date'].iloc[0].date()} to {test_data['date'].iloc[-1].date()}")
    print(f"BTC: ${test_data['close'].iloc[0]:,.0f} -> ${test_data['close'].iloc[-1]:,.0f}")
    print(f"\nAgent:      ${test_pnl:+,.0f} ± ${test_std:.0f}")
    print(f"Buy & Hold: ${test_bh:+,.0f}")

    diff = test_pnl - test_bh
    if diff > 0:
        print(f"\n✓ Agent WINS by ${diff:+,.0f}!")
    else:
        print(f"\nB&H wins by ${-diff:+,.0f}")

    os.remove(train_csv)
    algo.stop()
    ray.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    main()
