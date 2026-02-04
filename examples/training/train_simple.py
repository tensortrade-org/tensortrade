#!/usr/bin/env python3
"""
Simple training demo showing actual wallet balances and trade execution.
"""

import numpy as np
import pandas as pd
from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.env.default.actions import BSH
from tensortrade.env.default.rewards import PBR
import tensortrade.env.default as default


def main():
    print("="*70)
    print("TensorTrade Training Demo - Showing Wallet Balances")
    print("="*70)

    # Fetch data
    print("\nFetching BTC/USD data...")
    cdd = CryptoDataDownload()
    data = cdd.fetch("Bitfinex", "USD", "BTC", "1h")
    data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
    data = data.tail(200).reset_index(drop=True)
    print(f"Using {len(data)} rows | Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")

    # Setup
    print("\n" + "="*70)
    print("Creating Environment...")
    print("="*70)

    price_data = list(data["close"])
    price = Stream.source(price_data, dtype="float").rename("USD-BTC")

    exchange_options = ExchangeOptions(commission=0.001)
    exchange = Exchange("exchange", service=execute_order, options=exchange_options)(price)

    initial_cash = 10000
    cash = Wallet(exchange, initial_cash * USD)
    asset = Wallet(exchange, 0 * BTC)
    portfolio = Portfolio(USD, [cash, asset])

    # Features
    features = [Stream.source(list(data[c]), dtype="float").rename(c)
                for c in ['open', 'high', 'low', 'close', 'volume']]
    feed = DataFeed(features)
    feed.compile()

    # PBR reward scheme works with BSH
    reward_scheme = PBR(price=price)
    action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme)

    env = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        window_size=5,
        max_allowed_loss=0.5
    )

    print(f"Initial Cash:  ${cash.balance.as_float():,.2f} USD")
    print(f"Initial Asset: {asset.balance.as_float():.8f} BTC")

    # Run episodes
    num_episodes = 5
    action_names = {0: "BUY ", 1: "SELL", 2: "HOLD"}

    for episode in range(num_episodes):
        # Reset for new episode
        price = Stream.source(price_data, dtype="float").rename("USD-BTC")
        exchange = Exchange("exchange", service=execute_order, options=exchange_options)(price)
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
            window_size=5,
            max_allowed_loss=0.5
        )

        obs, info = env.reset()
        done = False
        truncated = False
        step = 0
        total_reward = 0
        trades = []

        print(f"\n{'='*70}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*70}")
        print(f"{'Step':>5} | {'Action':>5} | {'USD Balance':>14} | {'BTC Balance':>14} | {'Net Worth':>12} | {'Reward':>10}")
        print("-"*70)

        initial_worth = portfolio.net_worth

        while not done and not truncated and step < 150:  # Limit steps for readability
            # Simple policy: random with bias towards holding
            rand = np.random.random()
            if rand < 0.15:
                action = 0  # Buy
            elif rand < 0.30:
                action = 1  # Sell
            else:
                action = 2  # Hold

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            usd_bal = cash.balance.as_float()
            btc_bal = asset.balance.as_float()
            worth = portfolio.net_worth

            # Log trades and periodic status
            if action != 2 or step == 1 or step % 30 == 0:
                print(f"{step:5d} | {action_names[action]} | ${usd_bal:>12,.2f} | {btc_bal:>13.6f} | ${worth:>10,.2f} | {reward:>+10.2f}")

            if action != 2:
                trades.append({"step": step, "action": action, "worth": worth})

        final_worth = portfolio.net_worth
        pnl = final_worth - initial_worth
        pnl_pct = (pnl / initial_worth) * 100

        print("-"*70)
        print(f"Episode Summary:")
        print(f"  Steps: {step} | Trades: {len(trades)} | Total Reward: {total_reward:.2f}")
        print(f"  Initial: ${initial_worth:,.2f} -> Final: ${final_worth:,.2f}")
        print(f"  P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
