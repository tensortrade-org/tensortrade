
import operator

from tensortrade.wallets import Wallet
from tensortrade.instruments import Quantity, USD, BTC


from tensortrade.exchanges.services.execution.simulated import execute_order
from tensortrade.exchanges import Exchange
from tensortrade.data.internal import create_wallet_source
from tensortrade.data import DataFeed, Stream, Reduce


def test_exchange_with_wallets_feed():

    ex1 = Exchange("coinbase", service=execute_order)(
        Stream([7000, 7500, 8300]).rename("USD-BTC"),
        Stream([200, 212, 400]).rename("USD-ETH")
    )

    ex2 = Exchange("binance", service=execute_order)(
        Stream([7005, 7600, 8200]).rename("USD-BTC"),
        Stream([201, 208, 402]).rename("USD-ETH"),
        Stream([56, 52, 60]).rename("USD-LTC")
    )

    wallet_btc = Wallet(ex1, 10 * BTC)
    wallet_btc_ds = create_wallet_source(wallet_btc)

    wallet_usd = Wallet(ex2, 1000 * USD)
    wallet_usd.withdraw(
        quantity=400 * USD,
        reason="test"
    )
    wallet_usd.deposit(
        quantity=Quantity(USD, 400, path_id="fake_id"),
        reason="test"
    )
    wallet_usd_ds = create_wallet_source(wallet_usd, include_worth=False)

    feed = DataFeed([ex1, ex2, wallet_btc_ds, wallet_usd_ds])

    assert feed.next() == {
        "coinbase:/USD-BTC": 7000,
        "coinbase:/USD-ETH": 200,
        "coinbase:/BTC:/free": 10,
        "coinbase:/BTC:/locked": 0,
        "coinbase:/BTC:/total": 10,
        "coinbase:/BTC:/worth": 70000,
        "binance:/USD-BTC": 7005,
        "binance:/USD-ETH": 201,
        "binance:/USD-LTC": 56,
        "binance:/USD:/free": 600,
        "binance:/USD:/locked": 400,
        "binance:/USD:/total": 1000
    }
