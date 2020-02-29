
import ssl
import pytest

ssl._create_default_https_context = ssl._create_unverified_context

from tensortrade.actions import ManagedRiskOrders
from tensortrade.utils import CryptoDataDownload
from tensortrade.wallets import Portfolio, Wallet
from tensortrade.exchanges import Exchange
from tensortrade.exchanges.services.execution.simulated import execute_order
from tensortrade.data import Stream
from tensortrade.instruments import USD, BTC, ETH, LTC
from tensortrade.environments import TradingEnvironment


def test_smoke():
    cdd = CryptoDataDownload()
    coinbase_btc = cdd.fetch("Coinbase", "USD", "BTC", "1h")
    coinbase_eth = cdd.fetch("Coinbase", "USD", "ETH", "1h")

    bitstamp_btc = cdd.fetch("Bitstamp", "USD", "BTC", "1h")
    bitstamp_eth = cdd.fetch("Bitstamp", "USD", "ETH", "1h")
    bitstamp_ltc = cdd.fetch("Bitstamp", "USD", "LTC", "1h")

    steps = 20
    coinbase = Exchange("coinbase", service=execute_order)(
        Stream("USD-BTC", list(coinbase_btc['close'][-steps:])),
        Stream("USD-ETH", list(coinbase_eth['close'][-steps:]))
    )

    bitstamp = Exchange("bitstamp", service=execute_order)(
        Stream("USD-BTC", list(bitstamp_btc['close'][-steps:])),
        Stream("USD-ETH", list(bitstamp_eth['close'][-steps:])),
        Stream("USD-LTC", list(bitstamp_ltc['close'][-steps:]))
    )

    portfolio = Portfolio(USD, [
        Wallet(coinbase, 200000 * USD),
        Wallet(coinbase, 0 * BTC),
    ])

    action_scheme = ManagedRiskOrders(
        durations=[4],
        stop_loss_percentages=[0.01],
        take_profit_percentages=[0.01],
        trade_sizes=[0.1]
    )

    env = TradingEnvironment(
        action_scheme=action_scheme,
        reward_scheme="simple",
        portfolio=portfolio
    )

    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

    portfolio.ledger.as_frame().to_clipboard(index=False)

    pytest.fail("Failed.")
