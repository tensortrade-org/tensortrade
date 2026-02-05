#!/usr/bin/env python3
"""
Train on historical BTC data with technical indicators, evaluate on recent prices.
Uses all available historical data for training, then tests on recent market.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("Warning: pandas_ta not installed, using basic indicators")

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

    if HAS_PANDAS_TA:
        # RSI - Relative Strength Index
        df['rsi'] = ta.rsi(df['close'], length=14)

        # MACD - Moving Average Convergence Divergence
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df['macd'] = macd.iloc[:, 0]  # MACD line
            df['macd_signal'] = macd.iloc[:, 1]  # Signal line
            df['macd_hist'] = macd.iloc[:, 2]  # Histogram

        # Bollinger Bands
        bbands = ta.bbands(df['close'], length=20, std=2)
        if bbands is not None:
            df['bb_upper'] = bbands.iloc[:, 0]
            df['bb_mid'] = bbands.iloc[:, 1]
            df['bb_lower'] = bbands.iloc[:, 2]
            df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Moving Averages
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['ema_12'] = ta.ema(df['close'], length=12)
        df['ema_26'] = ta.ema(df['close'], length=26)

        # ATR - Average True Range (volatility)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

        # Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None:
            df['stoch_k'] = stoch.iloc[:, 0]
            df['stoch_d'] = stoch.iloc[:, 1]

        # ADX - Average Directional Index (trend strength)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None:
            df['adx'] = adx.iloc[:, 0]

        # OBV - On Balance Volume
        df['obv'] = ta.obv(df['close'], df['volume'])

        # Rate of Change
        df['roc'] = ta.roc(df['close'], length=10)

    else:
        # Basic indicators without pandas_ta
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # EMA
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

        # Rate of Change
        df['roc'] = df['close'].pct_change(periods=10) * 100

    # Normalize price-based indicators relative to close price
    df['sma_20_pct'] = (df['close'] - df['sma_20']) / df['sma_20'] * 100
    df['sma_50_pct'] = (df['close'] - df['sma_50']) / df['sma_50'] * 100
    df['atr_pct'] = df['atr'] / df['close'] * 100

    # Volume features
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Price momentum
    df['momentum_1h'] = df['close'].pct_change(1) * 100
    df['momentum_4h'] = df['close'].pct_change(4) * 100
    df['momentum_24h'] = df['close'].pct_change(24) * 100

    # Volatility
    df['volatility'] = df['close'].rolling(window=24).std() / df['close'] * 100

    # Fill NaN values
    df = df.bfill().ffill()

    return df


class WalletTrackingCallbacks(DefaultCallbacks):
    """Track wallet/portfolio values at episode boundaries."""

    def on_episode_start(
        self,
        *,
        worker: Any,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: EpisodeV2,
        env_index: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        env = base_env.get_sub_environments()[env_index]
        if hasattr(env, 'portfolio'):
            episode.user_data["initial_net_worth"] = float(env.portfolio.net_worth)

    def on_episode_end(
        self,
        *,
        worker: Any,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: EpisodeV2,
        env_index: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        env = base_env.get_sub_environments()[env_index]
        if hasattr(env, 'portfolio'):
            final_worth = float(env.portfolio.net_worth)
            initial_worth = episode.user_data.get("initial_net_worth", 10000)
            pnl = final_worth - initial_worth
            episode.custom_metrics["final_net_worth"] = final_worth
            episode.custom_metrics["pnl"] = pnl
            episode.custom_metrics["pnl_pct"] = (pnl / initial_worth) * 100 if initial_worth > 0 else 0


def create_env(config: Dict[str, Any]):
    """Create TensorTrade environment with technical indicators."""
    data = pd.read_csv(config["csv_filename"], parse_dates=['date'])
    data = data.bfill().ffill()

    price = Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
    exchange = Exchange("exchange", service=execute_order,
                       options=ExchangeOptions(commission=0.001))(price)

    initial_cash = config.get("initial_cash", 10000)
    cash = Wallet(exchange, initial_cash * USD)
    asset = Wallet(exchange, 0 * BTC)
    portfolio = Portfolio(USD, [cash, asset])

    # Get feature columns (everything except date)
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
        window_size=config.get("window_size", 10),
        max_allowed_loss=config.get("max_allowed_loss", 0.5)
    )
    env.portfolio = portfolio
    return env


def main():
    print("=" * 70)
    print("TensorTrade - Train on History, Evaluate on Recent Prices")
    print("=" * 70)

    # Fetch ALL available historical data
    print("\nFetching all available BTC/USD historical data...")
    cdd = CryptoDataDownload()
    data = cdd.fetch("Bitfinex", "USD", "BTC", "1h")
    data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values(by='date', ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)

    print(f"\nTotal data available: {len(data):,} hourly candles")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")

    # Add technical indicators
    print("\nAdding technical indicators...")
    data = add_technical_indicators(data)

    # Get all numeric feature columns (excluding date)
    feature_cols = [col for col in data.columns if col != 'date' and data[col].dtype in ['float64', 'int64']]
    print(f"Features: {len(feature_cols)} indicators")
    print(f"  {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")

    # Split: Use last 30 days for evaluation, rest for training
    # This gives us evaluation on the most recent "today's prices"
    eval_days = 30
    eval_candles = eval_days * 24  # 720 hourly candles

    train_data = data.iloc[:-eval_candles].copy()
    eval_data = data.iloc[-eval_candles:].copy()

    print(f"\n{'='*70}")
    print("Data Split")
    print(f"{'='*70}")
    print(f"Training data:   {len(train_data):,} candles")
    print(f"  From: {train_data['date'].min()}")
    print(f"  To:   {train_data['date'].max()}")
    print(f"  Price range: ${train_data['close'].min():,.0f} - ${train_data['close'].max():,.0f}")
    print(f"\nEvaluation data: {len(eval_data):,} candles")
    print(f"  From: {eval_data['date'].min()}")
    print(f"  To:   {eval_data['date'].max()}")
    print(f"  Price range: ${eval_data['close'].min():,.0f} - ${eval_data['close'].max():,.0f}")

    # For training, use a rolling window approach with recent historical data
    # Use last 2000 candles before cutoff for faster training (~83 days)
    train_sample = train_data.tail(2000).reset_index(drop=True)
    print(f"\nUsing last {len(train_sample)} candles for training")
    print(f"  From: {train_sample['date'].iloc[0]}")
    print(f"  To:   {train_sample['date'].iloc[-1]}")

    # Save datasets
    train_csv = os.path.join(os.getcwd(), 'train_historical.csv')
    eval_csv = os.path.join(os.getcwd(), 'eval_recent.csv')
    train_sample.to_csv(train_csv, index=False)
    eval_data.reset_index(drop=True).to_csv(eval_csv, index=False)

    # Initialize Ray
    ray.init(num_cpus=8, ignore_reinit_error=True, log_to_driver=False)
    register_env("TradingEnv", create_env)

    print(f"\n{'='*70}")
    print("Phase 1: Training on Historical Data with Technical Indicators")
    print(f"{'='*70}")

    train_config = {
        "window_size": 20,  # Larger window to capture indicator patterns
        "max_allowed_loss": 0.5,
        "csv_filename": train_csv,
        "initial_cash": 10000,
        "feature_cols": feature_cols,
    }

    # More extensive training config
    num_iterations = 100  # More iterations for better learning

    config = (
        PPOConfig().api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="TradingEnv", env_config=train_config)
        .framework("torch")
        .env_runners(num_env_runners=6)  # More parallel workers
        .callbacks(WalletTrackingCallbacks)
        .training(
            lr=1e-4,  # Lower learning rate for stability
            gamma=0.995,  # Higher gamma for longer-term rewards
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.005,  # Less exploration as we train longer
            train_batch_size=12000,  # Larger batches
            minibatch_size=512,  # Larger minibatches
            num_epochs=15,  # More SGD iterations per batch
            vf_clip_param=1000.0,
        )
        .resources(num_gpus=0)
    )

    algo = config.build()

    print(f"\nTraining for {num_iterations} iterations with {len(feature_cols)} features...")
    print("-" * 80)
    print(f"{'Iter':>4} | {'Avg Reward':>12} | {'Avg Net Worth':>14} | {'Avg P&L':>12} | {'Status':>12}")
    print("-" * 80)

    import time
    start_time = time.time()
    best_reward = float('-inf')
    rewards_history = []

    for i in range(num_iterations):
        result = algo.train()

        hist = result.get('env_runners', {})
        ep_reward_mean = hist.get('episode_return_mean')
        custom = hist.get('custom_metrics', {})
        final_net_worth_mean = custom.get('final_net_worth_mean', 0)
        pnl_mean = custom.get('pnl_mean', 0)

        if ep_reward_mean is not None:
            rewards_history.append(ep_reward_mean)
            marker = "*BEST*" if ep_reward_mean > best_reward else ""
            if ep_reward_mean > best_reward:
                best_reward = ep_reward_mean

            net_worth_str = f"${final_net_worth_mean:,.0f}" if final_net_worth_mean else "N/A"
            pnl_str = f"${pnl_mean:+,.0f}" if pnl_mean else "N/A"

            # Only print every 5 iterations to reduce noise, or if new best
            if (i + 1) % 5 == 0 or marker or i < 5:
                elapsed = time.time() - start_time
                print(f"{i+1:4d} | {ep_reward_mean:>+12,.0f} | {net_worth_str:>14} | {pnl_str:>12} | {marker:>12}")

    train_time = time.time() - start_time
    print("-" * 80)
    print(f"Training completed in {train_time/60:.1f} minutes")

    # Show training progress summary
    if len(rewards_history) >= 10:
        first_10 = np.mean(rewards_history[:10])
        last_10 = np.mean(rewards_history[-10:])
        improvement = last_10 - first_10
        print(f"  First 10 iterations avg reward: {first_10:+,.0f}")
        print(f"  Last 10 iterations avg reward:  {last_10:+,.0f}")
        print(f"  Improvement: {improvement:+,.0f} ({improvement/abs(first_10)*100 if first_10 != 0 else 0:+.1f}%)")

    # Phase 2: Evaluate on recent data (most recent 30 days)
    print(f"\n{'='*70}")
    print("Phase 2: Evaluating on Recent Data (Current Prices)")
    print(f"{'='*70}")

    eval_config = {
        "window_size": 20,
        "max_allowed_loss": 0.5,
        "csv_filename": eval_csv,
        "initial_cash": 10000,
        "feature_cols": feature_cols,
    }

    # Update algo config for evaluation
    algo.config.environment(env_config=eval_config)

    print(f"\nRunning evaluation episodes on recent prices...")
    print(f"Price range: ${eval_data['close'].min():,.0f} - ${eval_data['close'].max():,.0f}")
    print("-" * 70)

    # Run evaluation episodes (more episodes for statistical significance)
    num_eval_episodes = 20
    eval_results = []
    for ep in range(num_eval_episodes):
        # Create fresh eval environment
        eval_env = create_env(eval_config)
        obs, info = eval_env.reset()

        done = False
        truncated = False
        total_reward = 0
        steps = 0
        initial_worth = eval_env.portfolio.net_worth

        while not done and not truncated:
            action = algo.compute_single_action(obs)
            obs, reward, done, truncated, info = eval_env.step(action)
            total_reward += reward
            steps += 1

        final_worth = eval_env.portfolio.net_worth
        pnl = final_worth - initial_worth
        pnl_pct = (pnl / initial_worth) * 100

        eval_results.append({
            'episode': ep + 1,
            'steps': steps,
            'reward': total_reward,
            'final_worth': final_worth,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        })

        print(f"Episode {ep+1:2d} | Steps: {steps:4d} | Reward: {total_reward:>+10,.0f} | "
              f"Net Worth: ${final_worth:>10,.2f} | P&L: ${pnl:>+8,.2f} ({pnl_pct:>+6.2f}%)")

    algo.stop()

    # Summary
    print("-" * 70)
    print(f"\n{'='*70}")
    print("Final Results Summary")
    print(f"{'='*70}")

    avg_reward = np.mean([r['reward'] for r in eval_results])
    avg_pnl = np.mean([r['pnl'] for r in eval_results])
    avg_pnl_pct = np.mean([r['pnl_pct'] for r in eval_results])
    best_pnl = max([r['pnl'] for r in eval_results])
    worst_pnl = min([r['pnl'] for r in eval_results])

    print(f"\nTraining Period: {train_sample['date'].iloc[0].date()} to {train_sample['date'].iloc[-1].date()}")
    print(f"Evaluation Period: {eval_data['date'].iloc[0].date()} to {eval_data['date'].iloc[-1].date()}")
    print(f"\nEvaluation Results ({num_eval_episodes} episodes on recent prices):")
    print(f"  Initial Capital:  $10,000")
    print(f"  Avg Final Worth:  ${np.mean([r['final_worth'] for r in eval_results]):,.2f}")
    print(f"  Avg P&L:          ${avg_pnl:+,.2f} ({avg_pnl_pct:+.2f}%)")
    print(f"  Best P&L:         ${best_pnl:+,.2f}")
    print(f"  Worst P&L:        ${worst_pnl:+,.2f}")
    print(f"  Avg Reward:       {avg_reward:+,.0f}")

    # Compare to buy-and-hold
    start_price = eval_data['close'].iloc[0]
    end_price = eval_data['close'].iloc[-1]
    bh_return = ((end_price - start_price) / start_price) * 100
    bh_pnl = 10000 * (end_price - start_price) / start_price

    print(f"\nBuy-and-Hold Comparison (Evaluation Period):")
    print(f"  BTC Price: ${start_price:,.0f} -> ${end_price:,.0f}")
    print(f"  B&H Return: {bh_return:+.2f}%")
    print(f"  B&H P&L:    ${bh_pnl:+,.2f}")

    if avg_pnl > bh_pnl:
        print(f"\n  AGENT OUTPERFORMED buy-and-hold by ${avg_pnl - bh_pnl:+,.2f}")
    else:
        print(f"\n  Buy-and-hold outperformed agent by ${bh_pnl - avg_pnl:+,.2f}")

    ray.shutdown()

    # Cleanup temp files
    os.remove(train_csv)
    os.remove(eval_csv)

    print("\nDone!")


if __name__ == "__main__":
    main()
