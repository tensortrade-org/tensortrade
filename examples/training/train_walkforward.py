#!/usr/bin/env python3
"""
Walk-Forward Training with Multiple Market Regimes.

Best practices for RL trading:
1. Walk-forward validation - train on rolling windows, test on next period
2. Exposure to multiple market regimes (bull, bear, sideways)
3. Incremental learning - checkpoint and continue training
4. Risk-adjusted rewards consideration
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List

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


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to OHLCV dataframe."""
    df = df.copy()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Moving Averages
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
    df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()

    # Normalized features (scale-invariant)
    df['price_sma10_pct'] = (df['close'] - df['sma_10']) / df['sma_10'] * 100
    df['price_sma20_pct'] = (df['close'] - df['sma_20']) / df['sma_20'] * 100
    df['price_sma50_pct'] = (df['close'] - df['sma_50']) / df['sma_50'] * 100
    df['atr_pct'] = df['atr'] / df['close'] * 100

    # Volume features
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Momentum at multiple scales
    df['ret_1h'] = df['close'].pct_change(1) * 100
    df['ret_4h'] = df['close'].pct_change(4) * 100
    df['ret_12h'] = df['close'].pct_change(12) * 100
    df['ret_24h'] = df['close'].pct_change(24) * 100
    df['ret_72h'] = df['close'].pct_change(72) * 100

    # Volatility
    df['volatility_24h'] = df['close'].rolling(window=24).std() / df['close'] * 100
    df['volatility_72h'] = df['close'].rolling(window=72).std() / df['close'] * 100

    # Trend strength
    df['trend'] = (df['sma_10'] - df['sma_50']) / df['sma_50'] * 100

    # High/Low position
    df['high_low_pct'] = (df['close'] - df['low']) / (df['high'] - df['low'])

    df = df.bfill().ffill()
    return df


def identify_market_regime(df: pd.DataFrame, window: int = 72) -> pd.Series:
    """Identify market regime: bull (1), bear (-1), sideways (0)."""
    returns = df['close'].pct_change(window) * 100
    volatility = df['close'].rolling(window).std() / df['close'] * 100

    regime = pd.Series(0, index=df.index)
    regime[returns > 5] = 1   # Bull: >5% gain over window
    regime[returns < -5] = -1  # Bear: >5% loss over window
    # Sideways: between -5% and +5%

    return regime


class WalletTrackingCallbacks(DefaultCallbacks):
    """Track wallet/portfolio values at episode boundaries."""

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        env = base_env.get_sub_environments()[env_index]
        if hasattr(env, 'portfolio'):
            episode.user_data["initial_net_worth"] = float(env.portfolio.net_worth)

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        env = base_env.get_sub_environments()[env_index]
        if hasattr(env, 'portfolio'):
            final_worth = float(env.portfolio.net_worth)
            initial_worth = episode.user_data.get("initial_net_worth", 10000)
            pnl = final_worth - initial_worth
            pnl_pct = (pnl / initial_worth) * 100 if initial_worth > 0 else 0
            episode.custom_metrics["final_net_worth"] = final_worth
            episode.custom_metrics["pnl"] = pnl
            episode.custom_metrics["pnl_pct"] = pnl_pct


def create_env(config: Dict[str, Any]):
    """Create TensorTrade environment."""
    data = pd.read_csv(config["csv_filename"], parse_dates=['date'])
    data = data.bfill().ffill()

    price = Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
    exchange = Exchange("exchange", service=execute_order,
                       options=ExchangeOptions(commission=0.001))(price)

    initial_cash = config.get("initial_cash", 10000)
    cash = Wallet(exchange, initial_cash * USD)
    asset = Wallet(exchange, 0 * BTC)
    portfolio = Portfolio(USD, [cash, asset])

    feature_cols = config.get("feature_cols", ['open', 'high', 'low', 'close', 'volume'])
    features = []
    for col in feature_cols:
        if col in data.columns:
            features.append(Stream.source(list(data[col]), dtype="float").rename(col))

    feed = DataFeed(features)
    feed.compile()

    reward_scheme = PBR(price=price)
    action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme)

    env = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        window_size=config.get("window_size", 20),
        max_allowed_loss=config.get("max_allowed_loss", 0.6)
    )
    env.portfolio = portfolio
    return env


def evaluate_agent(algo, eval_config: Dict, num_episodes: int = 10) -> Dict:
    """Evaluate agent on test data."""
    results = []

    for _ in range(num_episodes):
        env = create_env(eval_config)
        obs, _ = env.reset()
        done = truncated = False
        total_reward = 0
        initial_worth = env.portfolio.net_worth

        while not done and not truncated:
            action = algo.compute_single_action(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward

        final_worth = env.portfolio.net_worth
        pnl = final_worth - initial_worth
        results.append({
            'reward': total_reward,
            'final_worth': final_worth,
            'pnl': pnl,
            'pnl_pct': (pnl / initial_worth) * 100
        })

    return {
        'avg_reward': np.mean([r['reward'] for r in results]),
        'avg_pnl': np.mean([r['pnl'] for r in results]),
        'avg_pnl_pct': np.mean([r['pnl_pct'] for r in results]),
        'best_pnl': max([r['pnl'] for r in results]),
        'worst_pnl': min([r['pnl'] for r in results]),
        'results': results
    }


def main():
    print("=" * 80)
    print("TensorTrade - Walk-Forward Training with Multiple Market Regimes")
    print("=" * 80)

    # Fetch ALL available historical data
    print("\nFetching all available BTC/USD historical data...")
    cdd = CryptoDataDownload()
    data = cdd.fetch("Bitfinex", "USD", "BTC", "1h")
    data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values(by='date', ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)

    print(f"Total data: {len(data):,} hourly candles")
    print(f"Date range: {data['date'].min().date()} to {data['date'].max().date()}")
    print(f"Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")

    # Add technical indicators
    print("\nAdding technical indicators...")
    data = add_technical_indicators(data)

    # Identify market regimes
    data['regime'] = identify_market_regime(data)
    bull_pct = (data['regime'] == 1).mean() * 100
    bear_pct = (data['regime'] == -1).mean() * 100
    sideways_pct = (data['regime'] == 0).mean() * 100
    print(f"Market regimes: Bull {bull_pct:.1f}% | Bear {bear_pct:.1f}% | Sideways {sideways_pct:.1f}%")

    feature_cols = [col for col in data.columns if col not in ['date', 'regime'] and data[col].dtype in ['float64', 'int64']]
    print(f"Features: {len(feature_cols)} indicators")

    # Walk-forward setup: Create multiple training windows
    # Use last 6 months, split into training folds
    total_hours = len(data)
    eval_hours = 30 * 24  # 30 days for final evaluation
    fold_hours = 45 * 24  # 45 days per training fold

    # Reserve last 30 days for final evaluation
    eval_data = data.iloc[-eval_hours:].copy()
    train_pool = data.iloc[:-eval_hours].copy()

    print(f"\n{'='*80}")
    print("Walk-Forward Training Setup")
    print(f"{'='*80}")
    print(f"Training pool: {len(train_pool):,} candles ({len(train_pool)//24} days)")
    print(f"Final evaluation: {len(eval_data):,} candles ({len(eval_data)//24} days)")

    # Create diverse training folds from different time periods
    # Sample from: recent (last 3 months), mid (3-6 months ago), and older data
    folds = []

    # Recent period (last 3 months before eval)
    recent_end = len(train_pool)
    recent_start = max(0, recent_end - 90 * 24)
    folds.append(('Recent (0-3mo)', train_pool.iloc[recent_start:recent_end].copy()))

    # Mid period (3-6 months ago)
    mid_end = recent_start
    mid_start = max(0, mid_end - 90 * 24)
    if mid_start < mid_end:
        folds.append(('Mid (3-6mo)', train_pool.iloc[mid_start:mid_end].copy()))

    # Older period (6-12 months ago) - sample for variety
    old_end = mid_start
    old_start = max(0, old_end - 180 * 24)
    if old_start < old_end:
        folds.append(('Older (6-12mo)', train_pool.iloc[old_start:old_end].copy()))

    # Bull market sample (if available)
    bull_data = train_pool[train_pool['regime'] == 1]
    if len(bull_data) > fold_hours:
        # Take a contiguous segment
        bull_segments = []
        start_idx = None
        for i, (idx, row) in enumerate(bull_data.iterrows()):
            if start_idx is None:
                start_idx = idx
            if i == len(bull_data) - 1 or bull_data.index[i+1] - idx > 24:
                if idx - start_idx > fold_hours // 2:
                    bull_segments.append((start_idx, idx))
                start_idx = None
        if bull_segments:
            best_seg = max(bull_segments, key=lambda x: x[1] - x[0])
            folds.append(('Bull Market', train_pool.loc[best_seg[0]:best_seg[1]].head(fold_hours).copy()))

    # Bear market sample (if available)
    bear_data = train_pool[train_pool['regime'] == -1]
    if len(bear_data) > fold_hours // 2:
        bear_segments = []
        start_idx = None
        for i, (idx, row) in enumerate(bear_data.iterrows()):
            if start_idx is None:
                start_idx = idx
            if i == len(bear_data) - 1 or bear_data.index[i+1] - idx > 24:
                if idx - start_idx > fold_hours // 4:
                    bear_segments.append((start_idx, idx))
                start_idx = None
        if bear_segments:
            best_seg = max(bear_segments, key=lambda x: x[1] - x[0])
            folds.append(('Bear Market', train_pool.loc[best_seg[0]:best_seg[1]].head(fold_hours).copy()))

    print(f"\nTraining folds: {len(folds)}")
    for name, fold_data in folds:
        regime_counts = fold_data['regime'].value_counts()
        print(f"  {name}: {len(fold_data):,} candles, "
              f"Bull:{regime_counts.get(1,0)} Bear:{regime_counts.get(-1,0)} Sideways:{regime_counts.get(0,0)}")

    # Initialize Ray
    ray.init(num_cpus=8, ignore_reinit_error=True, log_to_driver=False)
    register_env("TradingEnv", create_env)

    # Save initial training data (first fold) so algorithm can be built
    initial_fold_csv = os.path.join(os.getcwd(), 'initial_fold.csv')
    folds[0][1].reset_index(drop=True).to_csv(initial_fold_csv, index=False)

    # Create algorithm with initial data
    base_config = {
        "window_size": 20,
        "max_allowed_loss": 0.6,
        "initial_cash": 10000,
        "feature_cols": feature_cols,
        "csv_filename": initial_fold_csv,
    }

    config = (
        PPOConfig().api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="TradingEnv", env_config=base_config)
        .framework("torch")
        .env_runners(num_env_runners=6)
        .callbacks(WalletTrackingCallbacks)
        .training(
            lr=5e-5,  # Lower LR for stability across regimes
            gamma=0.995,
            lambda_=0.95,
            clip_param=0.15,  # Tighter clipping
            entropy_coeff=0.01,  # More exploration
            train_batch_size=16000,
            minibatch_size=512,
            num_epochs=20,
            vf_clip_param=1000.0,
        )
        .resources(num_gpus=0)
    )

    algo = config.build()

    import time
    start_time = time.time()

    # Combined training on diverse data
    print(f"\n{'='*80}")
    print("Phase 1: Training on Combined Diverse Market Data")
    print(f"{'='*80}")

    # Combine all folds into one diverse training set
    combined_data = pd.concat([f[1] for f in folds], ignore_index=True)
    print(f"\nCombined training data: {len(combined_data):,} candles from {len(folds)} market periods")

    # Save combined data
    combined_csv = os.path.join(os.getcwd(), 'combined_train.csv')
    combined_data.to_csv(combined_csv, index=False)

    # Update base config
    base_config['csv_filename'] = combined_csv

    total_iterations = 150  # More extensive training
    all_rewards = []

    print(f"\nTraining for {total_iterations} iterations...")
    print("-" * 80)
    print(f"{'Iter':>4} | {'Avg Reward':>12} | {'Net Worth':>12} | {'P&L':>12} | Status")
    print("-" * 80)

    best_reward = float('-inf')

    for i in range(total_iterations):
        result = algo.train()

        ep_reward = result.get('env_runners', {}).get('episode_return_mean')
        if ep_reward is not None:
            all_rewards.append(ep_reward)

            custom = result.get('env_runners', {}).get('custom_metrics', {})
            pnl = custom.get('pnl_mean', 0)
            net_worth = custom.get('final_net_worth_mean', 0)

            marker = "*BEST*" if ep_reward > best_reward else ""
            if ep_reward > best_reward:
                best_reward = ep_reward

            # Print every 5 iterations or if new best
            if (i + 1) % 5 == 0 or marker or i < 5:
                nw_str = f"${net_worth:,.0f}" if net_worth else "N/A"
                pnl_str = f"${pnl:+,.0f}" if pnl else "N/A"
                print(f"{i+1:4d} | {ep_reward:>+12,.0f} | {nw_str:>12} | {pnl_str:>12} | {marker}")

    os.remove(combined_csv)
    os.remove(initial_fold_csv)

    train_time = time.time() - start_time
    print("-" * 80)
    print(f"\nTraining Complete: {len(all_rewards)} iterations in {train_time/60:.1f} minutes")

    if len(all_rewards) >= 20:
        first_20 = np.mean(all_rewards[:20])
        last_20 = np.mean(all_rewards[-20:])
        print(f"  First 20 iterations avg: {first_20:+,.0f}")
        print(f"  Last 20 iterations avg:  {last_20:+,.0f}")
        print(f"  Improvement: {last_20 - first_20:+,.0f}")

    # Phase 2: Final Evaluation
    print(f"\n{'='*80}")
    print("Phase 2: Final Evaluation on Held-Out Recent Data")
    print(f"{'='*80}")

    eval_csv = os.path.join(os.getcwd(), 'eval_final.csv')
    eval_data.reset_index(drop=True).to_csv(eval_csv, index=False)

    eval_config = base_config.copy()
    eval_config['csv_filename'] = eval_csv

    print(f"\nEvaluation period: {eval_data['date'].iloc[0].date()} to {eval_data['date'].iloc[-1].date()}")
    print(f"Price range: ${eval_data['close'].min():,.0f} - ${eval_data['close'].max():,.0f}")
    print(f"Running 30 evaluation episodes...\n")

    eval_results = evaluate_agent(algo, eval_config, num_episodes=30)

    print(f"{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"\nEvaluation Performance (30 episodes):")
    print(f"  Initial Capital:   $10,000")
    print(f"  Avg Final Worth:   ${np.mean([r['final_worth'] for r in eval_results['results']]):,.2f}")
    print(f"  Avg P&L:           ${eval_results['avg_pnl']:+,.2f} ({eval_results['avg_pnl_pct']:+.2f}%)")
    print(f"  Best P&L:          ${eval_results['best_pnl']:+,.2f}")
    print(f"  Worst P&L:         ${eval_results['worst_pnl']:+,.2f}")
    print(f"  Avg Reward:        {eval_results['avg_reward']:+,.0f}")

    # Buy-and-hold comparison
    start_price = eval_data['close'].iloc[0]
    end_price = eval_data['close'].iloc[-1]
    bh_return = ((end_price - start_price) / start_price) * 100
    bh_pnl = 10000 * (end_price - start_price) / start_price

    print(f"\nBuy-and-Hold Benchmark:")
    print(f"  BTC Price: ${start_price:,.0f} -> ${end_price:,.0f}")
    print(f"  B&H Return: {bh_return:+.2f}%")
    print(f"  B&H P&L:    ${bh_pnl:+,.2f}")

    diff = eval_results['avg_pnl'] - bh_pnl
    if diff > 0:
        print(f"\n  ✓ AGENT OUTPERFORMED buy-and-hold by ${diff:+,.2f}!")
    else:
        print(f"\n  Buy-and-hold outperformed by ${-diff:+,.2f}")
        print(f"  (Agent showed {eval_results['avg_pnl_pct']:+.2f}% vs B&H {bh_return:+.2f}%)")

    # Win rate
    wins = sum(1 for r in eval_results['results'] if r['pnl'] > bh_pnl)
    print(f"\n  Win rate vs B&H: {wins}/{len(eval_results['results'])} episodes ({wins/len(eval_results['results'])*100:.0f}%)")

    # Cleanup
    os.remove(eval_csv)
    algo.stop()
    ray.shutdown()

    print("\nDone!")


if __name__ == "__main__":
    main()
