#!/usr/bin/env python3
"""
Ray RLlib PPO training script for TensorTrade.
Based on examples/use_lstm_rllib.ipynb

Run with: python run_ray_simulation.py
"""

import os
import numpy as np
import pandas as pd
import ta

# ============== Helper Functions ==============

def rsi(price: pd.Series, period: float) -> pd.Series:
    """Calculate RSI indicator."""
    r = price.diff()
    upside = np.minimum(r, 0).abs()
    downside = np.maximum(r, 0).abs()
    rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
    return 100 * (1 - (1 + rs) ** -1)


def macd(price: pd.Series, fast: float, slow: float, signal: float) -> pd.Series:
    """Calculate MACD indicator."""
    fm = price.ewm(span=fast, adjust=False).mean()
    sm = price.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    signal_line = md - md.ewm(span=signal, adjust=False).mean()
    return signal_line


# ============== Data Preparation ==============

def prepare_data():
    """Fetch and prepare data, save to CSV files."""
    from tensortrade.data.cdd import CryptoDataDownload
    from sklearn.model_selection import train_test_split

    print("Fetching crypto data...")
    cdd = CryptoDataDownload()
    data = cdd.fetch("Bitfinex", "USD", "BTC", "1h")
    data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
    data['volume'] = np.int64(data['volume'])
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values(by='date', ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Use last 5000 rows for faster training
    data = data.tail(5000).reset_index(drop=True)
    print(f"Using {len(data)} rows of data")

    # Split data
    X_train_test, X_valid, _, _ = train_test_split(
        data, data['close'].pct_change(),
        train_size=0.67, test_size=0.33, shuffle=False
    )
    X_train, X_test, _, _ = train_test_split(
        X_train_test, X_train_test['close'].pct_change(),
        train_size=0.50, test_size=0.50, shuffle=False
    )

    # Save to CSV
    cwd = os.getcwd()
    train_csv = os.path.join(cwd, 'train.csv')
    test_csv = os.path.join(cwd, 'test.csv')
    valid_csv = os.path.join(cwd, 'valid.csv')

    X_train.to_csv(train_csv, index=False)
    X_test.to_csv(test_csv, index=False)
    X_valid.to_csv(valid_csv, index=False)

    print(f"Saved train ({len(X_train)}), test ({len(X_test)}), valid ({len(X_valid)}) CSVs")
    return train_csv, test_csv, valid_csv


# ============== Environment Factory ==============

def create_env(config):
    """Create TensorTrade environment - called by Ray workers."""
    import tensortrade.env.default as default
    from tensortrade.env.default.rewards import PBR
    from tensortrade.env.default.actions import BSH
    from tensortrade.feed.core import DataFeed, Stream
    from tensortrade.feed.core.base import NameSpace
    from tensortrade.oms.exchanges import Exchange, ExchangeOptions
    from tensortrade.oms.instruments import USD, BTC
    from tensortrade.oms.services.execution.simulated import execute_order
    from tensortrade.oms.wallets import Wallet, Portfolio

    # Read data from CSV
    data = pd.read_csv(
        filepath_or_buffer=config["csv_filename"],
        parse_dates=['date']
    ).bfill().ffill()

    # Setup exchange
    commission = 0.001
    price = Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
    bitstamp_options = ExchangeOptions(commission=commission)
    bitstamp = Exchange("bitstamp", service=execute_order, options=bitstamp_options)(price)

    # Setup wallets and portfolio
    cash = Wallet(bitstamp, 100000 * USD)
    asset = Wallet(bitstamp, 0 * BTC)
    portfolio = Portfolio(USD, [cash, asset])

    # Generate features
    features = pd.DataFrame.from_dict({
        'close': data['close'],
        'volume': data['volume'],
        'dfast': data['close'].rolling(window=10).std().abs(),
        'dmedium': data['close'].rolling(window=50).std().abs(),
        'fast': data['close'].rolling(window=10).mean(),
        'medium': data['close'].rolling(window=50).mean(),
        'lr': np.log(data['close']).diff().fillna(0),
        'rsi_7': rsi(data['close'], period=7),
        'rsi_14': rsi(data['close'], period=14),
        'macd_normal': macd(data['close'], fast=12, slow=26, signal=9),
    }).bfill().ffill()

    # Create feed streams
    with NameSpace("bitstamp"):
        feature_streams = [
            Stream.source(list(features[c]), dtype="float").rename(c)
            for c in features.columns
        ]

    feed = DataFeed(feature_streams)
    feed.compile()

    # Setup reward and action schemes
    reward_scheme = PBR(price=price)
    action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme)

    # Renderer feed
    renderer_feed = DataFeed([
        Stream.source(list(data["date"])).rename("date"),
        Stream.source(list(data["open"]), dtype="float").rename("open"),
        Stream.source(list(data["high"]), dtype="float").rename("high"),
        Stream.source(list(data["low"]), dtype="float").rename("low"),
        Stream.source(list(data["close"]), dtype="float").rename("close"),
        Stream.source(list(data["volume"]), dtype="float").rename("volume"),
        Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action")
    ])

    # Create environment (omit renderer to use default EmptyRenderer)
    environment = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        renderer_feed=renderer_feed,
        window_size=config.get("window_size", 14),
        max_allowed_loss=config.get("max_allowed_loss", 0.9)
    )

    return environment


# ============== Main Training ==============

def main():
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    print("=" * 60)
    print("TensorTrade Ray RLlib Training")
    print("=" * 60)

    # Prepare data
    train_csv, test_csv, valid_csv = prepare_data()

    # Initialize Ray
    ray.init(
        num_cpus=4,
        ignore_reinit_error=True,
        log_to_driver=False
    )
    print("Ray initialized")

    # Register environment
    register_env("TradingEnv", create_env)
    print("Environment registered")

    # Training config
    env_config_training = {
        "window_size": 14,
        "max_allowed_loss": 0.5,
        "csv_filename": train_csv
    }

    env_config_evaluation = {
        "max_allowed_loss": 1.0,
        "csv_filename": test_csv,
    }

    print("\nStarting PPO training...")

    try:
        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(env="TradingEnv", env_config=env_config_training)
            .framework("torch")
            .env_runners(num_env_runners=2)
            .training(
                lr=1e-4,
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.2,
                entropy_coeff=0.01,
                train_batch_size=2000,
                minibatch_size=128,
                num_epochs=10,
            )
            .resources(num_gpus=0)
            .evaluation(
                evaluation_interval=2,
                evaluation_config={"env_config": env_config_evaluation, "explore": False},
            )
        )

        algo = config.build()

        num_iterations = 10
        print(f"\nTraining for {num_iterations} iterations...")

        best_reward = float('-inf')
        for i in range(num_iterations):
            result = algo.train()
            reward = result.get("env_runners", {}).get("episode_reward_mean", 0)
            print(f"  Iteration {i+1}/{num_iterations} | "
                  f"reward_mean={reward:.2f}")
            if reward > best_reward:
                best_reward = reward
            if reward >= 5000:
                print(f"  Target reward reached!")
                break

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best reward: {best_reward:.2f}")

        algo.stop()

    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
