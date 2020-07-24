from tensortrade.oms.wallets import Wallet
from tensortrade.oms.instruments import Quantity, USD, BTC


from tensortrade.oms.exchanges import execute_order
from tensortrade.oms.exchanges import Exchange
from tensortrade.data.internal import create_wallet_source
from tensortrade.data import DataFeed, Stream


def test_exchange_with_wallets_feed():

    ex1 = Exchange("coinbase", service=execute_order)(
        Stream.source([7000, 7500, 8300], dtype="float").rename("USD-BTC"),
        Stream.source([200, 212, 400], dtype="float").rename("USD-ETH")
    )

    ex2 = Exchange("binance", service=execute_order)(
        Stream.source([7005, 7600, 8200], dtype="float").rename("USD-BTC"),
        Stream.source([201, 208, 402], dtype="float").rename("USD-ETH"),
        Stream.source([56, 52, 60], dtype="float").rename("USD-LTC")
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

    streams = ex1.streams() + ex2.streams() + wallet_btc_ds + wallet_usd_ds
    feed = DataFeed(streams)

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
