#!/usr/bin/env python3
"""
Robust training with anti-overfitting techniques:
1. Normalized/scale-invariant features only
2. Higher entropy (more exploration)
3. Early stopping based on validation performance
4. Noise injection for robustness
5. Simpler model (smaller network)
6. Percentage-based returns (scale-invariant)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.env.default.actions import BSH
from tensortrade.env.default.rewards import PBR
import tensortrade.env.default as default


def add_scale_invariant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ONLY scale-invariant features (percentages, ratios, normalized values).
    This helps the model generalize across different price levels.
    """
    df = df.copy()

    # Returns at multiple timeframes (percentage-based, scale-invariant)
    for period in [1, 4, 12, 24, 48, 72, 168]:  # 1h to 1 week
        df[f'ret_{period}h'] = df['close'].pct_change(period) * 100
        df[f'ret_{period}h'] = df[f'ret_{period}h'].clip(-20, 20)  # Clip outliers

    # Volatility (normalized by price)
    for window in [12, 24, 72]:
        df[f'vol_{window}h'] = (df['close'].rolling(window).std() / df['close']) * 100

    # RSI (already 0-100 scale)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_norm'] = (df['rsi'] - 50) / 50  # Normalize to [-1, 1]

    # Price position relative to moving averages (percentage)
    for window in [10, 20, 50]:
        sma = df['close'].rolling(window).mean()
        df[f'price_sma{window}_pct'] = ((df['close'] - sma) / sma) * 100
        df[f'price_sma{window}_pct'] = df[f'price_sma{window}_pct'].clip(-15, 15)

    # Bollinger Band position (0-1 scale)
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
    df['bb_position'] = df['bb_position'].clip(0, 1)

    # MACD histogram normalized
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df['macd_hist_norm'] = ((macd - signal) / df['close']) * 100
    df['macd_hist_norm'] = df['macd_hist_norm'].clip(-2, 2)

    # Volume ratio (scale-invariant)
    vol_sma = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (vol_sma + 1e-10)
    df['volume_ratio'] = df['volume_ratio'].clip(0, 5)
    df['volume_ratio_log'] = np.log1p(df['volume_ratio'])

    # High-low range as percentage of close
    df['range_pct'] = ((df['high'] - df['low']) / df['close']) * 100

    # Trend strength (normalized)
    sma10 = df['close'].rolling(10).mean()
    sma50 = df['close'].rolling(50).mean()
    df['trend_strength'] = ((sma10 - sma50) / sma50) * 100
    df['trend_strength'] = df['trend_strength'].clip(-20, 20)

    # Momentum oscillator
    df['momentum'] = df['close'].pct_change(10) * 100
    df['momentum'] = df['momentum'].clip(-15, 15)

    # Add small noise to prevent overfitting to exact patterns
    noise_cols = [c for c in df.columns if c not in ['date', 'open', 'high', 'low', 'close', 'volume']]
    for col in noise_cols:
        noise = np.random.normal(0, 0.01, len(df))
        df[col] = df[col] + noise * df[col].std()

    df = df.bfill().ffill()
    return df


class WalletCallbacks(DefaultCallbacks):
    """Track performance metrics."""

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        env = base_env.get_sub_environments()[env_index]
        if hasattr(env, 'portfolio'):
            episode.user_data["initial_net_worth"] = float(env.portfolio.net_worth)

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        env = base_env.get_sub_environments()[env_index]
        if hasattr(env, 'portfolio'):
            final = float(env.portfolio.net_worth)
            initial = episode.user_data.get("initial_net_worth", 10000)
            pnl = final - initial
            episode.custom_metrics["final_net_worth"] = final
            episode.custom_metrics["pnl"] = pnl
            episode.custom_metrics["pnl_pct"] = (pnl / initial) * 100


def create_env(config: Dict[str, Any]):
    """Create environment with scale-invariant features."""
    data = pd.read_csv(config["csv_filename"], parse_dates=['date'])
    data = data.bfill().ffill()

    price = Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
    exchange = Exchange("exchange", service=execute_order,
                       options=ExchangeOptions(commission=0.001))(price)

    cash = Wallet(exchange, config.get("initial_cash", 10000) * USD)
    asset = Wallet(exchange, 0 * BTC)
    portfolio = Portfolio(USD, [cash, asset])

    feature_cols = config.get("feature_cols", [])
    features = [Stream.source(list(data[c]), dtype="float").rename(c)
                for c in feature_cols if c in data.columns]

    feed = DataFeed(features)
    feed.compile()

    reward_scheme = PBR(price=price)
    action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme)

    env = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        window_size=config.get("window_size", 15),
        max_allowed_loss=config.get("max_allowed_loss", 0.4)  # Tighter stop loss
    )
    env.portfolio = portfolio
    return env


def evaluate_on_data(algo, data: pd.DataFrame, feature_cols: list, num_episodes: int = 10) -> Dict:
    """Evaluate agent on specific data."""
    eval_csv = '/tmp/eval_temp.csv'
    data.reset_index(drop=True).to_csv(eval_csv, index=False)

    config = {
        "csv_filename": eval_csv,
        "feature_cols": feature_cols,
        "window_size": 15,
        "max_allowed_loss": 0.4,
        "initial_cash": 10000,
    }

    results = []
    for _ in range(num_episodes):
        env = create_env(config)
        obs, _ = env.reset()
        done = truncated = False
        total_reward = 0
        initial = env.portfolio.net_worth

        while not done and not truncated:
            action = algo.compute_single_action(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward

        final = env.portfolio.net_worth
        results.append({
            'pnl': final - initial,
            'pnl_pct': (final - initial) / initial * 100,
            'reward': total_reward
        })

    os.remove(eval_csv)

    return {
        'avg_pnl': np.mean([r['pnl'] for r in results]),
        'avg_pnl_pct': np.mean([r['pnl_pct'] for r in results]),
        'avg_reward': np.mean([r['reward'] for r in results]),
        'std_pnl': np.std([r['pnl'] for r in results]),
    }


def main():
    print("=" * 80)
    print("TensorTrade - Robust Training (Anti-Overfitting)")
    print("=" * 80)

    # Fetch data
    print("\nFetching BTC/USD data...")
    cdd = CryptoDataDownload()
    data = cdd.fetch("Bitfinex", "USD", "BTC", "1h")
    data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values('date', inplace=True)
    data.reset_index(drop=True, inplace=True)

    print(f"Total: {len(data):,} candles ({data['date'].min().date()} to {data['date'].max().date()})")

    # Add scale-invariant features
    print("\nAdding scale-invariant features (percentages, ratios only)...")
    data = add_scale_invariant_features(data)

    # Get only the normalized/percentage features (exclude raw OHLCV)
    feature_cols = [c for c in data.columns
                   if c not in ['date', 'open', 'high', 'low', 'close', 'volume']]
    print(f"Features: {len(feature_cols)} scale-invariant indicators")

    # Split data: Train on older, validate on middle, test on recent
    test_days = 30
    val_days = 30
    test_candles = test_days * 24
    val_candles = val_days * 24

    test_data = data.iloc[-test_candles:].copy()
    val_data = data.iloc[-(test_candles + val_candles):-test_candles].copy()
    train_data = data.iloc[:-(test_candles + val_candles)].copy()

    # Use last 4000 candles of training data (~167 days)
    train_data = train_data.tail(4000).reset_index(drop=True)

    print(f"\nData splits:")
    print(f"  Train: {len(train_data):,} candles ({train_data['date'].iloc[0].date()} to {train_data['date'].iloc[-1].date()})")
    print(f"  Val:   {len(val_data):,} candles ({val_data['date'].iloc[0].date()} to {val_data['date'].iloc[-1].date()})")
    print(f"  Test:  {len(test_data):,} candles ({test_data['date'].iloc[0].date()} to {test_data['date'].iloc[-1].date()})")

    # Save training data
    train_csv = os.path.join(os.getcwd(), 'train_robust.csv')
    train_data.to_csv(train_csv, index=False)

    # Initialize Ray
    ray.init(num_cpus=8, ignore_reinit_error=True, log_to_driver=False)
    register_env("TradingEnv", create_env)

    base_config = {
        "csv_filename": train_csv,
        "feature_cols": feature_cols,
        "window_size": 15,
        "max_allowed_loss": 0.4,
        "initial_cash": 10000,
    }

    # PPO config with anti-overfitting settings
    config = (
        PPOConfig().api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="TradingEnv", env_config=base_config)
        .framework("torch")
        .env_runners(num_env_runners=4)
        .callbacks(WalletCallbacks)
        .training(
            lr=1e-4,
            gamma=0.99,
            lambda_=0.9,  # Lower lambda for less credit assignment
            clip_param=0.1,  # Tighter clipping = smaller updates
            entropy_coeff=0.05,  # HIGH entropy = more exploration, less overfitting
            train_batch_size=8000,
            minibatch_size=256,
            num_epochs=5,  # Fewer SGD iterations = less overfitting
            vf_clip_param=100.0,
            kl_coeff=0.3,  # KL penalty for stability
            # Smaller network
            model={
                "fcnet_hiddens": [64, 64],  # Smaller than default [256, 256]
                "fcnet_activation": "tanh",
            },
        )
        .resources(num_gpus=0)
    )

    algo = config.build()

    print(f"\n{'='*80}")
    print("Training with Early Stopping on Validation Performance")
    print(f"{'='*80}")
    print(f"  Features: {len(feature_cols)} scale-invariant")
    print(f"  Network: [64, 64] (small)")
    print(f"  Entropy: 0.05 (high exploration)")
    print(f"  Clip: 0.1 (conservative updates)")
    print()

    import time
    start = time.time()

    max_iterations = 200
    patience = 20  # Stop if no improvement for 20 iterations
    best_val_pnl = float('-inf')
    best_iteration = 0
    no_improve_count = 0

    train_rewards = []
    val_pnls = []

    print("-" * 90)
    print(f"{'Iter':>4} | {'Train Reward':>12} | {'Train P&L':>10} | {'Val P&L':>10} | {'Val vs B&H':>10} | Status")
    print("-" * 90)

    # Buy-and-hold on validation
    val_start_price = val_data['close'].iloc[0]
    val_end_price = val_data['close'].iloc[-1]
    val_bh_pnl = 10000 * (val_end_price - val_start_price) / val_start_price

    for i in range(max_iterations):
        result = algo.train()

        train_reward = result.get('env_runners', {}).get('episode_return_mean', 0)
        train_pnl = result.get('env_runners', {}).get('custom_metrics', {}).get('pnl_mean', 0)
        train_rewards.append(train_reward)

        # Validate every 10 iterations
        if (i + 1) % 10 == 0:
            val_result = evaluate_on_data(algo, val_data, feature_cols, num_episodes=5)
            val_pnl = val_result['avg_pnl']
            val_pnls.append(val_pnl)

            vs_bh = val_pnl - val_bh_pnl

            if val_pnl > best_val_pnl:
                best_val_pnl = val_pnl
                best_iteration = i + 1
                no_improve_count = 0
                status = "*BEST*"
            else:
                no_improve_count += 1
                status = f"({no_improve_count}/{patience})"

            print(f"{i+1:4d} | {train_reward:>+12,.0f} | ${train_pnl:>+8,.0f} | ${val_pnl:>+8,.0f} | ${vs_bh:>+8,.0f} | {status}")

            # Early stopping
            if no_improve_count >= patience // 10:  # Check every 10 iters, so patience/10
                print(f"\nEarly stopping at iteration {i+1} (no improvement for {patience} iterations)")
                break
        else:
            # Print training progress
            if (i + 1) % 5 == 0 or i < 5:
                print(f"{i+1:4d} | {train_reward:>+12,.0f} | ${train_pnl:>+8,.0f} |            |            |")

    train_time = time.time() - start
    print("-" * 90)
    print(f"\nTraining stopped at iteration {i+1} ({train_time/60:.1f} minutes)")
    print(f"Best validation P&L: ${best_val_pnl:+,.0f} at iteration {best_iteration}")

    # Final test evaluation
    print(f"\n{'='*80}")
    print("Final Test Evaluation (Held-Out Data)")
    print(f"{'='*80}")

    test_result = evaluate_on_data(algo, test_data, feature_cols, num_episodes=30)

    # Buy-and-hold on test
    test_start = test_data['close'].iloc[0]
    test_end = test_data['close'].iloc[-1]
    test_bh_pnl = 10000 * (test_end - test_start) / test_start
    test_bh_pct = (test_end - test_start) / test_start * 100

    print(f"\nTest Period: {test_data['date'].iloc[0].date()} to {test_data['date'].iloc[-1].date()}")
    print(f"BTC Price: ${test_start:,.0f} -> ${test_end:,.0f} ({test_bh_pct:+.1f}%)")

    print(f"\nAgent Performance (30 episodes):")
    print(f"  Avg P&L:     ${test_result['avg_pnl']:+,.2f} ({test_result['avg_pnl_pct']:+.2f}%)")
    print(f"  Std P&L:     ${test_result['std_pnl']:,.2f}")
    print(f"  Avg Reward:  {test_result['avg_reward']:+,.0f}")

    print(f"\nBuy-and-Hold:")
    print(f"  P&L:         ${test_bh_pnl:+,.2f} ({test_bh_pct:+.2f}%)")

    diff = test_result['avg_pnl'] - test_bh_pnl
    if diff > 0:
        print(f"\n  ✓ AGENT OUTPERFORMED by ${diff:+,.2f}!")
    else:
        print(f"\n  Buy-and-hold won by ${-diff:+,.2f}")
        # Show relative performance
        if test_bh_pnl < 0:
            print(f"  (Agent lost {test_result['avg_pnl_pct']:.1f}% vs B&H lost {test_bh_pct:.1f}%)")
        else:
            print(f"  (Agent: {test_result['avg_pnl_pct']:+.1f}% vs B&H: {test_bh_pct:+.1f}%)")

    # Cleanup
    os.remove(train_csv)
    algo.stop()
    ray.shutdown()

    print("\nDone!")


if __name__ == "__main__":
    main()
