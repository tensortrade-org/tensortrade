#!/usr/bin/env python3
"""
TrendPBR sweep: tries multiple configurations to find positive test PnL.

The test period is a BTC downtrend, so the agent needs to learn to stay in
cash during downtrends. This script sweeps network size, entropy, commission,
and trend signal strength to find configs that generalize.

Usage:
    uv run python examples/training/train_trend_sweep.py
    uv run python examples/training/train_trend_sweep.py --configs trend-small pbr-baseline
    uv run python examples/training/train_trend_sweep.py --iters 30   # quick test
"""

import argparse
import os
import shutil
import sys
import pathlib
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# Ensure project root is on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tensortrade_platform.data.cdd import CryptoDataDownload
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.env.default.actions import BSH
from tensortrade.env.default.rewards import PBR, TrendPBR
import tensortrade.env.default as default


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

@dataclass
class SweepConfig:
    label: str
    reward: str        # "TrendPBR" or "PBR"
    hiddens: list[int]
    entropy: float
    commission: float
    trend_weight: float
    trend_scale: float


SWEEP_CONFIGS: dict[str, SweepConfig] = {
    "trend-small": SweepConfig(
        label="trend-small", reward="TrendPBR", hiddens=[64, 64],
        entropy=0.03, commission=0.003, trend_weight=0.001, trend_scale=20,
    ),
    "trend-medium": SweepConfig(
        label="trend-medium", reward="TrendPBR", hiddens=[128, 128],
        entropy=0.02, commission=0.003, trend_weight=0.001, trend_scale=20,
    ),
    "trend-deep": SweepConfig(
        label="trend-deep", reward="TrendPBR", hiddens=[128, 128, 64],
        entropy=0.02, commission=0.003, trend_weight=0.001, trend_scale=20,
    ),
    "trend-restraint": SweepConfig(
        label="trend-restraint", reward="TrendPBR", hiddens=[64, 64],
        entropy=0.05, commission=0.005, trend_weight=0.002, trend_scale=30,
    ),
    "pbr-baseline": SweepConfig(
        label="pbr-baseline", reward="PBR", hiddens=[64, 64],
        entropy=0.03, commission=0.003, trend_weight=0.0, trend_scale=0.0,
    ),
    "trend-tiny": SweepConfig(
        label="trend-tiny", reward="TrendPBR", hiddens=[32, 32],
        entropy=0.08, commission=0.003, trend_weight=0.001, trend_scale=20,
    ),
    # --- Round 2: generalization-focused configs ---
    "trend-cautious": SweepConfig(
        label="trend-cautious", reward="TrendPBR", hiddens=[64, 64],
        entropy=0.10, commission=0.005, trend_weight=0.003, trend_scale=40,
    ),
    "trend-med-ent": SweepConfig(
        label="trend-med-ent", reward="TrendPBR", hiddens=[128, 128],
        entropy=0.05, commission=0.003, trend_weight=0.001, trend_scale=20,
    ),
    "pbr-highent": SweepConfig(
        label="pbr-highent", reward="PBR", hiddens=[64, 64],
        entropy=0.10, commission=0.005, trend_weight=0.0, trend_scale=0.0,
    ),
    "trend-strong": SweepConfig(
        label="trend-strong", reward="TrendPBR", hiddens=[64, 64],
        entropy=0.05, commission=0.003, trend_weight=0.005, trend_scale=50,
    ),
    # --- Round 3: fine-tuning around trend-cautious sweet spot ---
    "cautious-v2": SweepConfig(
        label="cautious-v2", reward="TrendPBR", hiddens=[64, 64],
        entropy=0.12, commission=0.005, trend_weight=0.003, trend_scale=40,
    ),
    "cautious-v3": SweepConfig(
        label="cautious-v3", reward="TrendPBR", hiddens=[64, 64],
        entropy=0.10, commission=0.007, trend_weight=0.004, trend_scale=50,
    ),
    "cautious-v4": SweepConfig(
        label="cautious-v4", reward="TrendPBR", hiddens=[48, 48],
        entropy=0.10, commission=0.005, trend_weight=0.003, trend_scale=40,
    ),
    "cautious-v5": SweepConfig(
        label="cautious-v5", reward="TrendPBR", hiddens=[64, 64],
        entropy=0.10, commission=0.005, trend_weight=0.005, trend_scale=60,
    ),
    # --- Round 4: selective traders (catch upswings in downtrend) ---
    "selective-a": SweepConfig(
        label="selective-a", reward="TrendPBR", hiddens=[64, 64],
        entropy=0.04, commission=0.003, trend_weight=0.003, trend_scale=40,
    ),
    "selective-b": SweepConfig(
        label="selective-b", reward="TrendPBR", hiddens=[128, 128],
        entropy=0.04, commission=0.003, trend_weight=0.003, trend_scale=40,
    ),
    "selective-c": SweepConfig(
        label="selective-c", reward="TrendPBR", hiddens=[64, 64],
        entropy=0.06, commission=0.003, trend_weight=0.002, trend_scale=30,
    ),
    "selective-d": SweepConfig(
        label="selective-d", reward="TrendPBR", hiddens=[64, 64],
        entropy=0.04, commission=0.002, trend_weight=0.003, trend_scale=40,
    ),
    # --- Round 5: zero/low commission + more data ---
    "cautious-0comm": SweepConfig(
        label="cautious-0comm", reward="TrendPBR", hiddens=[64, 64],
        entropy=0.10, commission=0.0, trend_weight=0.003, trend_scale=40,
    ),
    "cautious-lowcomm": SweepConfig(
        label="cautious-lowcomm", reward="TrendPBR", hiddens=[64, 64],
        entropy=0.10, commission=0.001, trend_weight=0.003, trend_scale=40,
    ),
    "cautious-replay": SweepConfig(
        label="cautious-replay", reward="TrendPBR", hiddens=[64, 64],
        entropy=0.10, commission=0.005, trend_weight=0.003, trend_scale=40,
    ),
}


# ---------------------------------------------------------------------------
# Features (same as train_best.py)
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add scale-invariant features."""
    df = df.copy()

    for p in [1, 4, 12, 24, 48]:
        df[f'ret_{p}h'] = np.tanh(df['close'].pct_change(p) * 10)

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = (100 - (100 / (1 + rs)) - 50) / 50

    sma20 = df['close'].rolling(20).mean()
    sma50 = df['close'].rolling(50).mean()
    df['trend_20'] = np.tanh((df['close'] - sma20) / sma20 * 10)
    df['trend_50'] = np.tanh((df['close'] - sma50) / sma50 * 10)
    df['trend_strength'] = np.tanh((sma20 - sma50) / sma50 * 20)

    df['vol'] = df['close'].rolling(24).std() / df['close']
    df['vol_norm'] = np.tanh((df['vol'] - df['vol'].rolling(72).mean()) / df['vol'].rolling(72).std())

    df['vol_ratio'] = np.log1p(df['volume'] / df['volume'].rolling(20).mean())

    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_pos'] = ((df['close'] - (bb_mid - 2 * bb_std)) / (4 * bb_std)).clip(0, 1)

    return df.bfill().ffill()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class PnLCallbacks(DefaultCallbacks):
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


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def create_env(config: dict):
    """Create trading environment — dispatches on reward scheme."""
    data = pd.read_csv(config["csv_filename"], parse_dates=['date']).bfill().ffill()

    price = Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")

    commission = config.get("commission", 0.003)
    exchange = Exchange("exchange", service=execute_order,
                        options=ExchangeOptions(commission=commission))(price)

    cash = Wallet(exchange, 10000 * USD)
    asset = Wallet(exchange, 0 * BTC)
    portfolio = Portfolio(USD, [cash, asset])

    features = [Stream.source(list(data[c]), dtype="float").rename(c)
                for c in config.get("feature_cols", [])]
    feed = DataFeed(features)
    feed.compile()

    reward_name = config.get("reward_scheme", "PBR")

    if reward_name == "TrendPBR":
        reward_scheme = TrendPBR(
            price=price,
            commission=commission,
            trend_weight=config.get("trend_weight", 0.001),
            trend_scale=config.get("trend_scale", 20.0),
            trade_penalty_multiplier=1.0,
            churn_penalty_multiplier=0.75,
            churn_window=4,
        )
    else:
        reward_scheme = PBR(
            price=price,
            commission=commission,
            trade_penalty_multiplier=1.0,
            churn_penalty_multiplier=0.75,
            churn_window=4,
        )

    action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme)

    env = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        window_size=config.get("window_size", 10),
        max_allowed_loss=config.get("max_allowed_loss", 0.4),
    )
    env.portfolio = portfolio
    return env


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(algo, data: pd.DataFrame, feature_cols: list[str],
             env_config_base: dict, n: int = 30) -> float:
    """Evaluate on a data split and return average P&L."""
    csv = f'/tmp/eval_sweep_{os.getpid()}.csv'
    data.reset_index(drop=True).to_csv(csv, index=False)

    env_config = {**env_config_base, "csv_filename": csv}

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
    return float(np.mean(pnls))


# ---------------------------------------------------------------------------
# Single config training
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    label: str
    train_pnl: float
    val_pnl: float
    test_pnl: float
    test_pnl_zero: float
    best_iter: int
    elapsed_min: float


def train_config(
    cfg: SweepConfig,
    train_csv: str,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: list[str],
    max_iters: int,
    patience: int,
) -> SweepResult:
    """Train a single sweep configuration and return results."""
    print(f"\n{'='*70}")
    print(f"  Config: {cfg.label}")
    print(f"  Reward={cfg.reward}  Net={cfg.hiddens}  Entropy={cfg.entropy}")
    print(f"  Commission={cfg.commission}  TrendW={cfg.trend_weight}  TrendS={cfg.trend_scale}")
    print(f"{'='*70}")

    env_config = {
        "csv_filename": train_csv,
        "feature_cols": feature_cols,
        "window_size": 17,
        "max_allowed_loss": 0.32,
        "commission": cfg.commission,
        "reward_scheme": cfg.reward,
        "trend_weight": cfg.trend_weight,
        "trend_scale": cfg.trend_scale,
    }

    # Build eval config (same params, different data)
    eval_env_config = {
        k: v for k, v in env_config.items() if k != "csv_filename"
    }

    algo_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        .environment(env="TradingEnv", env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=2)
        .callbacks(PnLCallbacks)
        .training(
            lr=3.29e-05,
            gamma=0.992,
            lambda_=0.9,
            clip_param=0.123,
            entropy_coeff=cfg.entropy,
            vf_loss_coeff=0.5,
            train_batch_size=2000,
            minibatch_size=256,
            num_epochs=7,
            vf_clip_param=100.0,
            model={"fcnet_hiddens": cfg.hiddens, "fcnet_activation": "tanh"},
        )
        .resources(num_gpus=0)
    )

    algo = algo_config.build()
    checkpoint_dir = f'/tmp/sweep_best_{cfg.label}'

    best_val = float('-inf')
    best_iter = 0
    last_train_pnl = 0.0
    t0 = time.time()

    for i in range(max_iters):
        result = algo.train()

        if (i + 1) % 10 == 0:
            train_pnl = result.get('env_runners', {}).get(
                'custom_metrics', {}
            ).get('pnl_mean', 0)
            if train_pnl != train_pnl:  # NaN check
                train_pnl = 0.0
            last_train_pnl = train_pnl

            val_pnl = evaluate(algo, val_data, feature_cols, eval_env_config, n=10)

            marker = ""
            if val_pnl > best_val:
                best_val = val_pnl
                best_iter = i + 1
                marker = " *BEST*"
                if os.path.exists(checkpoint_dir):
                    shutil.rmtree(checkpoint_dir)
                algo.save(checkpoint_dir)

            print(f"  [{cfg.label}] Iter {i+1:3d}: "
                  f"Train ${train_pnl:+,.0f} | Val ${val_pnl:+,.0f} "
                  f"(best ${best_val:+,.0f} @{best_iter}){marker}")

            # Early stopping
            if (i + 1) - best_iter >= patience:
                print(f"  [{cfg.label}] Early stop at iter {i+1} "
                      f"(no improvement for {patience} iters)")
                break

    # Restore best checkpoint and evaluate on test
    if os.path.exists(checkpoint_dir):
        algo.restore(checkpoint_dir)

    test_pnl = evaluate(algo, test_data, feature_cols, eval_env_config, n=100)

    # Also evaluate with zero commission to see intrinsic trading skill
    zero_comm_config = {**eval_env_config, "commission": 0.0}
    test_pnl_zero = evaluate(algo, test_data, feature_cols, zero_comm_config, n=100)
    elapsed = (time.time() - t0) / 60

    print(f"\n  [{cfg.label}] RESULT: Train ${last_train_pnl:+,.0f} | "
          f"Val ${best_val:+,.0f} | Test ${test_pnl:+,.0f} "
          f"(0comm: ${test_pnl_zero:+,.0f}) ({elapsed:.1f} min)")

    # Cleanup
    algo.stop()
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    return SweepResult(
        label=cfg.label,
        train_pnl=last_train_pnl,
        val_pnl=best_val,
        test_pnl=test_pnl,
        test_pnl_zero=test_pnl_zero,
        best_iter=best_iter,
        elapsed_min=elapsed,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TrendPBR sweep for positive test PnL")
    parser.add_argument("--configs", nargs="+", default=list(SWEEP_CONFIGS.keys()),
                        choices=list(SWEEP_CONFIGS.keys()),
                        help="Which configs to run (default: all)")
    parser.add_argument("--iters", type=int, default=60,
                        help="Max training iterations per config (default: 60)")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stop patience in iterations (default: 30)")
    args = parser.parse_args()

    print("=" * 70)
    print("TrendPBR Sweep — Finding Positive Test PnL")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    cdd = CryptoDataDownload()
    data = cdd.fetch("Bitfinex", "BTC", "USD", "1h")
    data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values('date', inplace=True)
    data.reset_index(drop=True, inplace=True)

    print("Adding features...")
    data = add_features(data)
    feature_cols = [c for c in data.columns
                    if c not in ['date', 'open', 'high', 'low', 'close', 'volume']]

    # Split: train | val (30d) | test (30d)
    test_candles = 30 * 24   # 720h
    val_candles = 30 * 24    # 720h

    test_data = data.iloc[-test_candles:].copy()
    val_data = data.iloc[-(test_candles + val_candles):-test_candles].copy()
    train_data = data.iloc[:-(test_candles + val_candles)].tail(4000).reset_index(drop=True)

    val_bh = 10000 * (val_data['close'].iloc[-1] - val_data['close'].iloc[0]) / val_data['close'].iloc[0]
    test_bh = 10000 * (test_data['close'].iloc[-1] - test_data['close'].iloc[0]) / test_data['close'].iloc[0]

    print(f"\nData splits:")
    print(f"  Train: {len(train_data)} candles")
    print(f"  Val:   {len(val_data)} candles  (B&H ${val_bh:+,.0f})")
    print(f"  Test:  {len(test_data)} candles  (B&H ${test_bh:+,.0f})")
    print(f"  Test:  {test_data['date'].iloc[0].date()} to {test_data['date'].iloc[-1].date()}")
    print(f"  BTC:   ${test_data['close'].iloc[0]:,.0f} -> ${test_data['close'].iloc[-1]:,.0f}")

    # Save train CSV
    train_csv = f'/tmp/train_sweep_{os.getpid()}.csv'
    train_data.to_csv(train_csv, index=False)

    # Init Ray
    ray.init(num_cpus=6, ignore_reinit_error=True, log_to_driver=False)
    register_env("TradingEnv", create_env)

    # Run sweep
    configs_to_run = [SWEEP_CONFIGS[name] for name in args.configs]
    print(f"\nRunning {len(configs_to_run)} configs: {[c.label for c in configs_to_run]}")
    print(f"Max {args.iters} iters each, patience={args.patience}")

    results: list[SweepResult] = []
    for cfg in configs_to_run:
        result = train_config(
            cfg, train_csv, val_data, test_data, feature_cols,
            max_iters=args.iters, patience=args.patience,
        )
        results.append(result)

    # Cleanup
    os.remove(train_csv)
    ray.shutdown()

    # Summary table sorted by test PnL
    results.sort(key=lambda r: r.test_pnl, reverse=True)

    print(f"\n{'='*70}")
    print("SWEEP RESULTS (sorted by test PnL)")
    print(f"{'='*70}")
    print(f"{'Config':<18} {'Train':>10} {'Val':>10} {'Test':>10} {'Test0comm':>10} {'Best@':>6} {'Time':>6}")
    print(f"{'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*6}")

    for r in results:
        test_marker = " +" if r.test_pnl > 0 else ""
        zero_marker = " +" if r.test_pnl_zero > 0 else ""
        print(f"{r.label:<18} ${r.train_pnl:>+8,.0f} ${r.val_pnl:>+8,.0f} "
              f"${r.test_pnl:>+8,.0f}{test_marker} ${r.test_pnl_zero:>+8,.0f}{zero_marker} "
              f"{r.best_iter:>5}i {r.elapsed_min:>5.1f}m")

    print(f"\nBuy & Hold:  Val ${val_bh:+,.0f}  |  Test ${test_bh:+,.0f}")

    # Highlight winners
    positive = [r for r in results if r.test_pnl > 0]
    beats_bh = [r for r in results if r.test_pnl > test_bh]

    if positive:
        print(f"\n*** {len(positive)} config(s) with POSITIVE test PnL: "
              f"{[r.label for r in positive]} ***")
    else:
        print("\nNo configs achieved positive test PnL.")

    if beats_bh:
        print(f"*** {len(beats_bh)} config(s) BEAT buy-and-hold on test: "
              f"{[r.label for r in beats_bh]} ***")

    print("\nDone!")


if __name__ == "__main__":
    main()
