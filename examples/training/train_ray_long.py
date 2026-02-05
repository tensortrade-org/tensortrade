#!/usr/bin/env python3
"""
Long-running Ray RLlib training (5-10 minutes) with wallet balance tracking.
Shows portfolio net worth at episode boundaries via custom metrics.
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


class WalletTrackingCallbacks(DefaultCallbacks):
    """Callbacks to track wallet/portfolio values at episode boundaries."""

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
        """Called at the start of each episode."""
        # Get the underlying TensorTrade environment
        env = base_env.get_sub_environments()[env_index]

        # Access portfolio net worth
        if hasattr(env, 'portfolio'):
            initial_worth = float(env.portfolio.net_worth)
            episode.user_data["initial_net_worth"] = initial_worth
            episode.hist_data["initial_net_worth"] = [initial_worth]

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
        """Called at the end of each episode."""
        env = base_env.get_sub_environments()[env_index]

        if hasattr(env, 'portfolio'):
            final_worth = float(env.portfolio.net_worth)
            initial_worth = episode.user_data.get("initial_net_worth", 10000)

            # Calculate P&L
            pnl = final_worth - initial_worth
            pnl_pct = (pnl / initial_worth) * 100 if initial_worth > 0 else 0

            # Store in custom metrics
            episode.custom_metrics["final_net_worth"] = final_worth
            episode.custom_metrics["initial_net_worth"] = initial_worth
            episode.custom_metrics["pnl"] = pnl
            episode.custom_metrics["pnl_pct"] = pnl_pct
            episode.hist_data["final_net_worth"] = [final_worth]


def create_env(config: Dict[str, Any]):
    """Create TensorTrade environment with wallet tracking."""
    data = pd.read_csv(config["csv_filename"], parse_dates=['date'])
    data = data.fillna(method='backfill').fillna(method='ffill')

    price = Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
    exchange = Exchange("exchange", service=execute_order,
                       options=ExchangeOptions(commission=0.001))(price)

    initial_cash = config.get("initial_cash", 10000)
    cash = Wallet(exchange, initial_cash * USD)
    asset = Wallet(exchange, 0 * BTC)
    portfolio = Portfolio(USD, [cash, asset])

    features = [Stream.source(list(data[c]), dtype="float").rename(c)
                for c in ['open', 'high', 'low', 'close', 'volume']]
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

    # Store portfolio reference for callback access
    env.portfolio = portfolio

    return env


def main():
    print("=" * 70)
    print("TensorTrade Long Training Run (5-10 minutes)")
    print("With Wallet Balance Tracking")
    print("=" * 70)

    # Fetch longer historical data for extended training
    print("\nFetching BTC/USD historical data...")
    cdd = CryptoDataDownload()
    data = cdd.fetch("Bitfinex", "USD", "BTC", "1h")
    data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values(by='date', ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Use 500 rows = ~490 steps per episode (longer episodes)
    data = data.tail(500).reset_index(drop=True)

    print(f"Data: {len(data)} hourly candles (~21 days)")
    print(f"Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")
    print(f"Episode length: ~{len(data) - 10} steps")

    # Calculate price change over period
    price_start = data['close'].iloc[0]
    price_end = data['close'].iloc[-1]
    price_change_pct = ((price_end - price_start) / price_start) * 100
    print(f"Market move: ${price_start:,.0f} -> ${price_end:,.0f} ({price_change_pct:+.1f}%)")

    # Save to CSV
    train_csv = os.path.join(os.getcwd(), 'train.csv')
    data.to_csv(train_csv, index=False)

    # Initialize Ray with more resources
    ray.init(num_cpus=8, ignore_reinit_error=True, log_to_driver=False)
    register_env("TradingEnv", create_env)
    print(f"\nRay initialized with 8 CPUs")

    env_config = {
        "window_size": 10,
        "max_allowed_loss": 0.5,
        "csv_filename": train_csv,
        "initial_cash": 10000,
    }

    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"  Initial Cash:     $10,000")
    print(f"  Reward Scheme:    PBR (Position-Based Returns)")
    print(f"  Action Scheme:    BSH (Buy/Sell/Hold)")
    print(f"  Workers:          4 parallel environments")
    print(f"  Batch Size:       8000 steps")
    print(f"  Target Duration:  5-10 minutes")
    print(f"  Iterations:       100")
    print()

    config = (
        PPOConfig().api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="TradingEnv", env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=4)
        .callbacks(WalletTrackingCallbacks)
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            train_batch_size=8000,  # Larger batches for longer training
            minibatch_size=256,
            num_epochs=10,
            vf_clip_param=1000.0,
        )
        .resources(num_gpus=0)
    )

    algo = config.build()

    print("-" * 90)
    print(f"{'Iter':>4} | {'Episodes':>8} | {'Avg Reward':>12} | {'Avg Net Worth':>14} | {'Avg P&L':>12} | Progress")
    print("-" * 90)

    all_rewards = []
    all_net_worths = []
    all_pnls = []
    best_reward = float('-inf')

    import time
    start_time = time.time()

    for i in range(100):
        result = algo.train()

        elapsed = time.time() - start_time

        # Get episode rewards
        hist = result.get('env_runners', {})
        ep_reward_mean = hist.get('episode_return_mean')
        episodes_this_iter = hist.get('num_episodes', 0)

        # Get custom metrics (wallet tracking)
        custom = result.get('env_runners', {}).get('custom_metrics', {})
        final_net_worth_mean = custom.get('final_net_worth_mean', 0)
        pnl_mean = custom.get('pnl_mean', 0)
        pnl_pct_mean = custom.get('pnl_pct_mean', 0)

        if ep_reward_mean is not None:
            all_rewards.append(ep_reward_mean)

            if final_net_worth_mean:
                all_net_worths.append(final_net_worth_mean)
                all_pnls.append(pnl_mean)

            if ep_reward_mean > best_reward:
                best_reward = ep_reward_mean
                marker = " *BEST*"
            else:
                marker = ""

            # Visual bar based on reward
            bar_len = int((ep_reward_mean + 5000) / 500)
            bar_len = max(0, min(25, bar_len))
            bar = "#" * bar_len

            net_worth_str = f"${final_net_worth_mean:,.0f}" if final_net_worth_mean else "N/A"
            pnl_str = f"${pnl_mean:+,.0f}" if pnl_mean else "N/A"

            print(f"{i+1:4d} | {episodes_this_iter:>8} | {ep_reward_mean:>+12,.0f} | {net_worth_str:>14} | {pnl_str:>12} | {bar}{marker}")
        else:
            steps = result.get('num_env_steps_sampled_lifetime', 0)
            print(f"{i+1:4d} | Collecting... ({steps:,} steps, {elapsed:.0f}s)")

        # Stop after 10 minutes
        if elapsed > 600:
            print(f"\n[Reached 10 minute limit at iteration {i+1}]")
            break

    algo.stop()

    total_time = time.time() - start_time

    print("-" * 90)
    print("\n" + "=" * 70)
    print("Training Complete - Results Summary")
    print("=" * 70)

    print(f"\nTraining Duration: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    print(f"Iterations Completed: {len(all_rewards)}")

    if len(all_rewards) >= 6:
        # Divide into thirds for comparison
        third = len(all_rewards) // 3
        first_third = all_rewards[:third]
        last_third = all_rewards[-third:]

        first_avg = np.mean(first_third)
        last_avg = np.mean(last_third)
        improvement = last_avg - first_avg
        improvement_pct = (improvement / abs(first_avg)) * 100 if first_avg != 0 else 0

        print(f"\nReward Progression:")
        print(f"  First third avg:  {first_avg:>+12,.0f}")
        print(f"  Last third avg:   {last_avg:>+12,.0f}")
        print(f"  Improvement:      {improvement:>+12,.0f} ({improvement_pct:+.1f}%)")
        print(f"  Best reward:      {best_reward:>+12,.0f}")

        if improvement > 0:
            print(f"\n  CONFIRMED: Rewards improved by {improvement:+,.0f} over training!")
        else:
            print(f"\n  Training showed no improvement (may need different hyperparameters)")

    if all_net_worths:
        print(f"\nWallet Performance:")
        print(f"  Initial Cash:        $10,000")
        print(f"  Avg Final Net Worth: ${np.mean(all_net_worths):,.0f}")
        print(f"  Best Final Net Worth: ${max(all_net_worths):,.0f}")
        print(f"  Avg P&L:             ${np.mean(all_pnls):+,.0f}")

        # Show net worth trend
        if len(all_net_worths) >= 6:
            first_nw = np.mean(all_net_worths[:third])
            last_nw = np.mean(all_net_worths[-third:])
            nw_improvement = last_nw - first_nw
            print(f"\n  Net Worth Trend:")
            print(f"    Early episodes avg: ${first_nw:,.0f}")
            print(f"    Late episodes avg:  ${last_nw:,.0f}")
            print(f"    Change:             ${nw_improvement:+,.0f}")

    # Show reward trend visually
    if all_rewards:
        print(f"\nReward Trend (sampled every 5 iterations):")
        for i in range(0, len(all_rewards), 5):
            r = all_rewards[i]
            bar = "#" * max(0, min(40, int((r + 5000) / 400)))
            print(f"  {i+1:3d}: {bar} {r:+,.0f}")

    ray.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    main()
