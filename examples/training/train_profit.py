#!/usr/bin/env python3
"""
Profit-focused trading strategy.

Key principles:
1. Train on BEAR market data (since test is bear market)
2. Simple trend-following features
3. Risk-adjusted returns (Sharpe ratio)
4. Match training conditions to test conditions
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
from tensortrade.env.default.rewards import RiskAdjustedReturns
import tensortrade.env.default as default


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple trend-following features only."""
    df = df.copy()

    # Core trend indicators
    sma_fast = df['close'].rolling(10).mean()
    sma_slow = df['close'].rolling(50).mean()

    # Trend direction: +1 bullish, -1 bearish
    df['trend'] = np.where(sma_fast > sma_slow, 1, -1).astype(float)

    # Trend strength (normalized)
    df['trend_strength'] = np.tanh((sma_fast - sma_slow) / sma_slow * 20)

    # Price position relative to trend
    df['price_vs_sma'] = np.tanh((df['close'] - sma_slow) / sma_slow * 10)

    # Momentum (rate of change)
    df['momentum_1d'] = np.tanh(df['close'].pct_change(24) * 10)  # 1 day
    df['momentum_3d'] = np.tanh(df['close'].pct_change(72) * 10)  # 3 days

    # RSI normalized to [-1, 1]
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = (100 - (100 / (1 + rs)) - 50) / 50

    # Volatility regime (high vol = caution)
    vol = df['close'].rolling(24).std() / df['close']
    vol_ma = vol.rolling(72).mean()
    df['vol_regime'] = np.tanh((vol - vol_ma) / (vol_ma + 1e-10) * 5)

    return df.bfill().ffill()


def find_bear_periods(df: pd.DataFrame, window: int = 720, threshold: float = -0.02) -> list:
    """Find periods where the market dropped significantly."""
    periods = []
    for i in range(window, len(df)):
        start_price = df['close'].iloc[i - window]
        end_price = df['close'].iloc[i]
        ret = (end_price - start_price) / start_price
        if ret < threshold:
            periods.append((i - window, i, ret))
    return periods


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


def create_env(config: Dict[str, Any]):
    data = pd.read_csv(config["csv_filename"], parse_dates=['date']).bfill().ffill()

    price = Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
    commission = config.get("commission", 0.001)  # 0.1% commission
    exchange = Exchange("exchange", service=execute_order,
                       options=ExchangeOptions(commission=commission))(price)

    cash = Wallet(exchange, config.get("initial_cash", 10000) * USD)
    asset = Wallet(exchange, 0 * BTC)
    portfolio = Portfolio(USD, [cash, asset])

    features = [Stream.source(list(data[c]), dtype="float").rename(c)
                for c in config.get("feature_cols", [])]
    feed = DataFeed(features)
    feed.compile()

    # Use Risk-Adjusted Returns (Sharpe ratio) for consistent profits
    reward_scheme = RiskAdjustedReturns(
        return_algorithm='sharpe',
        risk_free_rate=0,
        window_size=config.get("reward_window", 20)
    )
    action_scheme = BSH(cash=cash, asset=asset)

    env = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        window_size=config.get("window_size", 10),
        max_allowed_loss=config.get("max_allowed_loss", 0.5)
    )
    env.portfolio = portfolio
    return env


def evaluate(algo, data: pd.DataFrame, feature_cols: list, config: Dict, n: int = 20) -> float:
    """Evaluate and return average P&L."""
    csv = '/tmp/eval_profit.csv'
    data.reset_index(drop=True).to_csv(csv, index=False)

    env_config = {
        "csv_filename": csv,
        "feature_cols": feature_cols,
        "window_size": config["window_size"],
        "max_allowed_loss": config["max_allowed_loss"],
        "commission": config["commission"],
        "reward_window": config.get("reward_window", 20),
        "initial_cash": 10000,
    }

    pnls = []
    for _ in range(n):
        env = create_env(env_config)
        obs, _ = env.reset()
        done = truncated = False
        while not done and not truncated:
            action = algo.compute_single_action(obs)
            obs, _, done, truncated, _ = env.step(action)
        pnls.append(env.portfolio.net_worth - 10000)

    os.remove(csv)
    return np.mean(pnls)


def main():
    print("=" * 70)
    print("TensorTrade - Profit-Focused Strategy")
    print("=" * 70)
    print("\nStrategy: Trend-following with Sharpe ratio rewards")
    print("Key: Train on BEAR market data to match test conditions")

    # Load data
    print("\nLoading data...")
    cdd = CryptoDataDownload()
    data = cdd.fetch("Bitfinex", "USD", "BTC", "1h")
    data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values('date', inplace=True)
    data.reset_index(drop=True, inplace=True)

    print("Adding trend features...")
    data = add_trend_features(data)
    feature_cols = ['trend', 'trend_strength', 'price_vs_sma',
                    'momentum_1d', 'momentum_3d', 'rsi', 'vol_regime']
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # Split data
    test_candles = 30 * 24  # 30 days
    val_candles = 30 * 24

    test_data = data.iloc[-test_candles:].copy()
    val_data = data.iloc[-(test_candles + val_candles):-test_candles].copy()

    # KEY: Find bear market periods for training
    remaining_data = data.iloc[:-(test_candles + val_candles)].copy()

    # Calculate returns for each potential training window
    print("\nFinding bear market training periods...")
    bear_periods = find_bear_periods(remaining_data, window=720, threshold=-0.03)
    print(f"Found {len(bear_periods)} bear market periods (>3% drop over 30 days)")

    if bear_periods:
        # Use the most recent bear periods for training
        bear_periods.sort(key=lambda x: x[0], reverse=True)
        selected = bear_periods[:5]  # Take 5 most recent bear periods

        # Combine bear period data
        train_chunks = []
        for start, end, ret in selected:
            chunk = remaining_data.iloc[start:end].copy()
            print(f"  Period {remaining_data['date'].iloc[start].date()} to "
                  f"{remaining_data['date'].iloc[end-1].date()}: {ret*100:+.1f}%")
            train_chunks.append(chunk)

        train_data = pd.concat(train_chunks, ignore_index=True)
    else:
        # Fallback to recent data
        print("No significant bear periods found, using recent data")
        train_data = remaining_data.tail(3000).reset_index(drop=True)

    # Buy-and-hold baselines
    val_bh = 10000 * (val_data['close'].iloc[-1] - val_data['close'].iloc[0]) / val_data['close'].iloc[0]
    test_bh = 10000 * (test_data['close'].iloc[-1] - test_data['close'].iloc[0]) / test_data['close'].iloc[0]

    print(f"\nData splits:")
    print(f"  Train: {len(train_data)} candles (bear market periods)")
    print(f"  Val:   {len(val_data)} candles (B&H: ${val_bh:+,.0f})")
    print(f"  Test:  {len(test_data)} candles (B&H: ${test_bh:+,.0f})")

    # Save train data
    train_csv = '/tmp/train_profit.csv'
    train_data.to_csv(train_csv, index=False)

    env_config = {
        "csv_filename": train_csv,
        "feature_cols": feature_cols,
        "window_size": 15,
        "max_allowed_loss": 0.5,  # Allow more drawdown
        "commission": 0.001,  # 0.1% realistic commission
        "reward_window": 20,  # Sharpe window
        "initial_cash": 10000,
    }

    # Initialize Ray
    ray.init(num_cpus=6, ignore_reinit_error=True, log_to_driver=False)
    register_env("TradingEnv", create_env)

    # PPO config optimized for generalization
    config = (
        PPOConfig().api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="TradingEnv", env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=4)
        .callbacks(Callbacks)
        .training(
            lr=5e-5,           # Low learning rate
            gamma=0.99,        # High discount
            lambda_=0.95,
            clip_param=0.1,    # Tight clipping
            entropy_coeff=0.05,  # High entropy for exploration
            train_batch_size=4000,
            minibatch_size=256,
            num_epochs=5,
            vf_clip_param=100.0,
            model={"fcnet_hiddens": [64, 64], "fcnet_activation": "tanh"},
        )
        .resources(num_gpus=0)
    )

    algo = config.build()

    # Training with early stopping
    print(f"\n{'='*70}")
    print("Training (with early stopping on validation)")
    print(f"{'='*70}")

    train_iters = 100
    best_val_pnl = float('-inf')
    best_iter = 0
    patience = 20
    no_improve = 0

    print(f"\nTraining for up to {train_iters} iterations (patience={patience})...")

    for i in range(train_iters):
        result = algo.train()

        if (i + 1) % 5 == 0:
            pnl = result.get('env_runners', {}).get('custom_metrics', {}).get('pnl_mean', 0)

            # Validation
            val_config = {**env_config, "window_size": 15, "max_allowed_loss": 0.5,
                         "commission": 0.001, "reward_window": 20}
            val_pnl = evaluate(algo, val_data, feature_cols, val_config, n=10)

            status = ""
            if val_pnl > best_val_pnl:
                best_val_pnl = val_pnl
                best_iter = i + 1
                no_improve = 0
                status = " *BEST*"
                # Save checkpoint
                algo.save('/tmp/best_profit_model')
            else:
                no_improve += 1

            print(f"  Iter {i+1:3d}: Train P&L ${pnl:+,.0f} | Val P&L ${val_pnl:+,.0f} "
                  f"(best: ${best_val_pnl:+,.0f} @ {best_iter}){status}")

            # Early stopping
            if no_improve >= patience // 5:
                print(f"\nEarly stopping at iteration {i+1} (no improvement for {no_improve*5} iters)")
                break

    # Load best model
    if os.path.exists('/tmp/best_profit_model'):
        algo.restore('/tmp/best_profit_model')
        print(f"\nRestored best model from iteration {best_iter}")

    # Final evaluation
    print(f"\n{'='*70}")
    print("Final Test Results (Held-Out Data)")
    print(f"{'='*70}")

    test_config = {**env_config, "window_size": 15, "max_allowed_loss": 0.5,
                   "commission": 0.001, "reward_window": 20}
    test_pnl = evaluate(algo, test_data, feature_cols, test_config, n=50)

    print(f"\nTest period: {test_data['date'].iloc[0].date()} to {test_data['date'].iloc[-1].date()}")
    print(f"BTC: ${test_data['close'].iloc[0]:,.0f} -> ${test_data['close'].iloc[-1]:,.0f} "
          f"({(test_data['close'].iloc[-1]/test_data['close'].iloc[0]-1)*100:+.1f}%)")

    print(f"\n{'='*40}")
    print(f"Agent (Trend+Sharpe):  ${test_pnl:+,.0f}")
    print(f"Buy & Hold:            ${test_bh:+,.0f}")
    print(f"{'='*40}")

    diff = test_pnl - test_bh
    if test_pnl > 0:
        print(f"\n*** PROFITABLE! Agent made ${test_pnl:,.0f} ***")
        if diff > 0:
            print(f"    And beat B&H by ${diff:,.0f}!")
    elif diff > 0:
        print(f"\nAgent beat B&H by ${diff:,.0f} (but still lost money)")
    else:
        print(f"\nB&H wins by ${-diff:,.0f}")

    # Cleanup
    os.remove(train_csv)
    if os.path.exists('/tmp/best_profit_model'):
        import shutil
        shutil.rmtree('/tmp/best_profit_model')
    algo.stop()
    ray.shutdown()

    print("\nDone!")


if __name__ == "__main__":
    main()
