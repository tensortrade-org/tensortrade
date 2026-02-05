#!/usr/bin/env python3
"""
Hyperparameter optimization with Optuna + Ray RLlib.

Runs multiple trials to find the best hyperparameters for trading.
Uses validation performance to guide the search, tests best on held-out data.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any
import optuna
from optuna.samplers import TPESampler

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

    reward_scheme = PBR(price=price)
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
    return env


def evaluate(algo, data: pd.DataFrame, feature_cols: list, config: Dict, n: int = 10) -> float:
    """Evaluate and return average P&L."""
    csv = '/tmp/eval_optuna.csv'
    data.reset_index(drop=True).to_csv(csv, index=False)

    env_config = {
        "csv_filename": csv,
        "feature_cols": feature_cols,
        "window_size": config["window_size"],
        "max_allowed_loss": config["max_allowed_loss"],
        "commission": config["commission"],
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


# Global data (loaded once)
TRAIN_DATA = None
VAL_DATA = None
TEST_DATA = None
FEATURE_COLS = None
VAL_BH = None
TEST_BH = None


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function - returns validation P&L."""
    global TRAIN_DATA, VAL_DATA, FEATURE_COLS

    # Hyperparameters to optimize
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    entropy = trial.suggest_float("entropy", 0.01, 0.2, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    clip = trial.suggest_float("clip", 0.05, 0.3)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    window_size = trial.suggest_int("window_size", 5, 20)
    max_loss = trial.suggest_float("max_loss", 0.2, 0.5)
    commission = trial.suggest_float("commission", 0.0001, 0.001, log=True)
    sgd_iters = trial.suggest_int("sgd_iters", 3, 15)
    batch_size = trial.suggest_categorical("batch_size", [2000, 4000, 8000])

    # Save train data
    train_csv = f'/tmp/train_optuna_{trial.number}.csv'
    TRAIN_DATA.to_csv(train_csv, index=False)

    env_config = {
        "csv_filename": train_csv,
        "feature_cols": FEATURE_COLS,
        "window_size": window_size,
        "max_allowed_loss": max_loss,
        "commission": commission,
        "initial_cash": 10000,
    }

    config = (
        PPOConfig().api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="TradingEnv", env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=2)  # Fewer workers for faster trials
        .callbacks(Callbacks)
        .training(
            lr=lr,
            gamma=gamma,
            lambda_=0.9,
            clip_param=clip,
            entropy_coeff=entropy,
            train_batch_size=batch_size,
            minibatch_size=min(256, batch_size // 4),
            num_epochs=sgd_iters,
            vf_clip_param=100.0,
            model={"fcnet_hiddens": [hidden_size, hidden_size], "fcnet_activation": "tanh"},
        )
        .resources(num_gpus=0)
    )

    algo = config.build()

    # Train for fixed iterations
    train_iters = 40
    for i in range(train_iters):
        algo.train()

        # Early pruning based on intermediate results
        if (i + 1) % 10 == 0:
            val_pnl = evaluate(algo, VAL_DATA, FEATURE_COLS,
                             {"window_size": window_size, "max_allowed_loss": max_loss,
                              "commission": commission}, n=5)
            trial.report(val_pnl, i)

            if trial.should_prune():
                algo.stop()
                os.remove(train_csv)
                raise optuna.TrialPruned()

    # Final validation
    val_config = {"window_size": window_size, "max_allowed_loss": max_loss, "commission": commission}
    val_pnl = evaluate(algo, VAL_DATA, FEATURE_COLS, val_config, n=10)

    algo.stop()
    os.remove(train_csv)

    return val_pnl


def main():
    global TRAIN_DATA, VAL_DATA, TEST_DATA, FEATURE_COLS, VAL_BH, TEST_BH

    print("=" * 70)
    print("TensorTrade - Optuna Hyperparameter Optimization")
    print("=" * 70)

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
    FEATURE_COLS = [c for c in data.columns if c not in ['date', 'open', 'high', 'low', 'close', 'volume']]
    print(f"Features: {len(FEATURE_COLS)}")

    # Split data
    test_candles = 30 * 24
    val_candles = 30 * 24

    TEST_DATA = data.iloc[-test_candles:].copy()
    VAL_DATA = data.iloc[-(test_candles + val_candles):-test_candles].copy()
    TRAIN_DATA = data.iloc[:-(test_candles + val_candles)].tail(3000).reset_index(drop=True)

    # Buy-and-hold baselines
    VAL_BH = 10000 * (VAL_DATA['close'].iloc[-1] - VAL_DATA['close'].iloc[0]) / VAL_DATA['close'].iloc[0]
    TEST_BH = 10000 * (TEST_DATA['close'].iloc[-1] - TEST_DATA['close'].iloc[0]) / TEST_DATA['close'].iloc[0]

    print(f"\nData splits:")
    print(f"  Train: {len(TRAIN_DATA)} candles")
    print(f"  Val:   {len(VAL_DATA)} candles (B&H: ${VAL_BH:+,.0f})")
    print(f"  Test:  {len(TEST_DATA)} candles (B&H: ${TEST_BH:+,.0f})")

    # Initialize Ray
    ray.init(num_cpus=6, ignore_reinit_error=True, log_to_driver=False)
    register_env("TradingEnv", create_env)

    # Create Optuna study
    print(f"\n{'='*70}")
    print("Starting Optuna Optimization")
    print(f"{'='*70}")

    n_trials = 100  # Number of different hyperparameter combinations to try

    study = optuna.create_study(
        direction="maximize",  # Maximize validation P&L
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    print(f"Running {n_trials} trials...")
    print(f"Each trial: 40 training iterations, evaluated on validation set\n")

    def callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"Trial {trial.number:2d}: Val P&L ${trial.value:+,.0f} | "
                  f"vs B&H ${trial.value - VAL_BH:+,.0f} | "
                  f"Best so far: ${study.best_value:+,.0f}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"Trial {trial.number:2d}: PRUNED (underperforming)")

    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)

    # Results
    print(f"\n{'='*70}")
    print("Optimization Complete")
    print(f"{'='*70}")

    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best validation P&L: ${study.best_value:+,.0f} (vs B&H ${VAL_BH:+,.0f})")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Train final model with best params and test
    print(f"\n{'='*70}")
    print("Training Final Model with Best Hyperparameters")
    print(f"{'='*70}")

    best = study.best_params
    train_csv = '/tmp/train_final.csv'
    TRAIN_DATA.to_csv(train_csv, index=False)

    env_config = {
        "csv_filename": train_csv,
        "feature_cols": FEATURE_COLS,
        "window_size": best["window_size"],
        "max_allowed_loss": best["max_loss"],
        "commission": best["commission"],
        "initial_cash": 10000,
    }

    final_config = (
        PPOConfig().api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="TradingEnv", env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=4)
        .callbacks(Callbacks)
        .training(
            lr=best["lr"],
            gamma=best["gamma"],
            lambda_=0.9,
            clip_param=best["clip"],
            entropy_coeff=best["entropy"],
            train_batch_size=best["batch_size"],
            minibatch_size=min(256, best["batch_size"] // 4),
            num_epochs=best["sgd_iters"],
            vf_clip_param=100.0,
            model={"fcnet_hiddens": [best["hidden_size"], best["hidden_size"]],
                   "fcnet_activation": "tanh"},
        )
        .resources(num_gpus=0)
    )

    final_algo = final_config.build()

    # Train longer with best params
    print("\nTraining final model for 60 iterations...")
    for i in range(60):
        result = final_algo.train()
        if (i + 1) % 15 == 0:
            pnl = result.get('env_runners', {}).get('custom_metrics', {}).get('pnl_mean', 0)
            print(f"  Iter {i+1}: Train P&L ${pnl:+,.0f}")

    # Final test evaluation
    print(f"\n{'='*70}")
    print("Final Test Results (Held-Out Data)")
    print(f"{'='*70}")

    test_config = {"window_size": best["window_size"], "max_allowed_loss": best["max_loss"],
                   "commission": best["commission"]}
    test_pnl = evaluate(final_algo, TEST_DATA, FEATURE_COLS, test_config, n=30)

    print(f"\nTest period: {TEST_DATA['date'].iloc[0].date()} to {TEST_DATA['date'].iloc[-1].date()}")
    print(f"BTC: ${TEST_DATA['close'].iloc[0]:,.0f} -> ${TEST_DATA['close'].iloc[-1]:,.0f}")

    print(f"\nAgent (optimized):  ${test_pnl:+,.0f}")
    print(f"Buy & Hold:         ${TEST_BH:+,.0f}")

    diff = test_pnl - TEST_BH
    if diff > 0:
        print(f"\n✓ AGENT WINS by ${diff:+,.0f}!")
    else:
        print(f"\nB&H wins by ${-diff:+,.0f}")

    # Show top 5 trials
    print(f"\n{'='*70}")
    print("Top 5 Trials Summary")
    print(f"{'='*70}")

    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df['state'] == 'COMPLETE'].sort_values('value', ascending=False)

    for i, (_, row) in enumerate(trials_df.head(5).iterrows()):
        print(f"{i+1}. Trial {int(row['number'])}: Val P&L ${row['value']:+,.0f} | "
              f"lr={row['params_lr']:.2e}, entropy={row['params_entropy']:.3f}, "
              f"hidden={int(row['params_hidden_size'])}")

    # Cleanup
    os.remove(train_csv)
    final_algo.stop()
    ray.shutdown()

    print("\nDone!")


if __name__ == "__main__":
    main()
