
from tensortrade.env.default.observers import _create_internal_streams, _create_wallet_source
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import USD, BTC, ETH, LTC, Quantity


def test_price_ds():

    btc_price = Stream.source([7000, 7500, 8300], dtype="float").rename("USD-BTC")
    eth_price = Stream.source([200, 212, 400], dtype="float").rename("USD-ETH")

    feed = DataFeed([btc_price, eth_price])

    assert feed.next() == {"USD-BTC": 7000, "USD-ETH": 200}


def test_exchange_feed():

    btc_price = Stream.source([7000, 7500, 8300], dtype="float").rename("USD-BTC")
    eth_price = Stream.source([200, 212, 400], dtype="float").rename("USD-ETH")

    exchange = Exchange("bitfinex", service=execute_order)(
        btc_price,
        eth_price
    )

    feed = DataFeed(exchange.streams())

    assert feed.next() == {"bitfinex:/USD-BTC": 7000, "bitfinex:/USD-ETH": 200}


def test_create_internal_data_feed():

    ex1 = Exchange("bitfinex", service=execute_order)(
        Stream.source([7000, 7500, 8300], dtype="float").rename("USD-BTC"),
        Stream.source([200, 212, 400], dtype="float").rename("USD-ETH")
    )

    ex2 = Exchange("binance", service=execute_order)(
        Stream.source([7005, 7600, 8200], dtype="float").rename("USD-BTC"),
        Stream.source([201, 208, 402], dtype="float").rename("USD-ETH"),
        Stream.source([56, 52, 60], dtype="float").rename("USD-LTC")
    )

    portfolio = Portfolio(USD, [
        Wallet(ex1, 10000 * USD),
        Wallet(ex1, 10 * BTC),
        Wallet(ex1, 5 * ETH),
        Wallet(ex2, 1000 * USD),
        Wallet(ex2, 5 * BTC),
        Wallet(ex2, 20 * ETH),
        Wallet(ex2, 3 * LTC),
    ])

    feed = DataFeed(_create_internal_streams(portfolio))

    data = {
        "bitfinex:/USD-BTC": 7000,
        "bitfinex:/USD-ETH": 200,
        "bitfinex:/USD:/free": 10000,
        "bitfinex:/USD:/locked": 0,
        "bitfinex:/USD:/total": 10000,
        "bitfinex:/BTC:/free": 10,
        "bitfinex:/BTC:/locked": 0,
        "bitfinex:/BTC:/total": 10,
        "bitfinex:/BTC:/worth": 7000 * 10,
        "bitfinex:/ETH:/free": 5,
        "bitfinex:/ETH:/locked": 0,
        "bitfinex:/ETH:/total": 5,
        "bitfinex:/ETH:/worth": 200 * 5,
        "binance:/USD-BTC": 7005,
        "binance:/USD-ETH": 201,
        "binance:/USD-LTC": 56,
        "binance:/USD:/free": 1000,
        "binance:/USD:/locked": 0,
        "binance:/USD:/total": 1000,
        "binance:/BTC:/free": 5,
        "binance:/BTC:/locked": 0,
        "binance:/BTC:/total": 5,
        "binance:/BTC:/worth": 7005 * 5,
        "binance:/ETH:/free": 20,
        "binance:/ETH:/locked": 0,
        "binance:/ETH:/total": 20,
        "binance:/ETH:/worth": 201 * 20,
        "binance:/LTC:/free": 3,
        "binance:/LTC:/locked": 0,
        "binance:/LTC:/total": 3,
        "binance:/LTC:/worth": 56 * 3,
    }

    bitfinex_net_worth = 10000 + (10 * 7000) + (5 * 200)
    binance_net_worth = 1000 + (5 * 7005) + (20 * 201) + (3 * 56)

    data['net_worth'] = sum(data[k] if k.endswith("worth") or k.endswith("USD:/total") else 0 for k in data.keys())

    assert feed.next() == data


def test_exchange_with_wallets_feed():

    ex1 = Exchange("bitfinex", service=execute_order)(
        Stream.source([7000, 7500, 8300], dtype="float").rename("USD-BTC"),
        Stream.source([200, 212, 400], dtype="float").rename("USD-ETH")
    )

    ex2 = Exchange("binance", service=execute_order)(
        Stream.source([7005, 7600, 8200], dtype="float").rename("USD-BTC"),
        Stream.source([201, 208, 402], dtype="float").rename("USD-ETH"),
        Stream.source([56, 52, 60], dtype="float").rename("USD-LTC")
    )

    wallet_btc = Wallet(ex1, 10 * BTC)
    wallet_btc_ds = _create_wallet_source(wallet_btc)

    wallet_usd = Wallet(ex2, 1000 * USD)
    wallet_usd.withdraw(
        quantity=400 * USD,
        reason="test"
    )
    wallet_usd.deposit(
        quantity=Quantity(USD, 400, path_id="fake_id"),
        reason="test"
    )
    wallet_usd_ds = _create_wallet_source(wallet_usd, include_worth=False)

    streams = ex1.streams() + ex2.streams() + wallet_btc_ds + wallet_usd_ds
    feed = DataFeed(streams)

    assert feed.next() == {
        "bitfinex:/USD-BTC": 7000,
        "bitfinex:/USD-ETH": 200,
        "bitfinex:/BTC:/free": 10,
        "bitfinex:/BTC:/locked": 0,
        "bitfinex:/BTC:/total": 10,
        "bitfinex:/BTC:/worth": 70000,
        "binance:/USD-BTC": 7005,
        "binance:/USD-ETH": 201,
        "binance:/USD-LTC": 56,
        "binance:/USD:/free": 600,
        "binance:/USD:/locked": 400,
        "binance:/USD:/total": 1000
    }
