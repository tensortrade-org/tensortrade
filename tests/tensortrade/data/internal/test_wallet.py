
import operator
import unittest.mock as mock

from tensortrade.wallets import Wallet
from tensortrade.instruments import Quantity, USD, BTC
from tensortrade.data.stream.transform import BinOp
from tensortrade.data.internal.wallet import Balance, LockedBalance
from tensortrade.data.stream.feed import DataFeed


def test_balance_ds():
    exchange = mock.Mock()
    exchange.name = "fake"
    wallet = Wallet(exchange, 10000*USD)

    balance = Balance(wallet)
    assert balance.next() == 10000

    wallet -= 1000 * USD
    wallet += Quantity(USD, 1000, path_id="fake_id")

    balance = Balance(wallet)
    assert balance.next() == 9000


def test_balance_feed():
    exchange = mock.Mock()
    exchange.name = "fake"
    w1 = Wallet(exchange, 10000 * USD)
    w2 = Wallet(exchange, 10 * BTC)

    w1 -= 1000 * USD
    w1 += Quantity(USD, 1000, path_id="fake_id")

    w2 -= 1 * BTC
    w2 += Quantity(BTC, 1, path_id="fake_id")

    b1 = Balance(w1)
    b2 = Balance(w2)

    feed = DataFeed([b1, b2])

    assert feed.next() == {"fake_USD": 9000, "fake_BTC": 9}


def test_locked_balance_ds():
    exchange = mock.Mock()
    exchange.name = "fake"
    wallet = Wallet(exchange, 10000*USD)

    locked_balance = LockedBalance(wallet)

    wallet -= 1000 * USD
    wallet += Quantity(USD, 1000, path_id="fake_id_0")

    wallet -= 2000 * USD
    wallet += Quantity(USD, 2000, path_id="fake_id_1")

    assert locked_balance.next() == 3000


def test_wallet_feed():

    exchange = mock.Mock()
    exchange.name = "fake"
    wallet = Wallet(exchange, 10000*USD)

    locked_balance = LockedBalance(wallet)
    balance = Balance(wallet)

    wallet -= 1000 * USD
    wallet += Quantity(USD, 1000, path_id="fake_id_0")

    wallet -= 2000 * USD
    wallet += Quantity(USD, 2000, path_id="fake_id_1")

    feed = DataFeed([balance, locked_balance])

    assert feed.next() == {"fake_USD": 7000, "fake_USD_locked": 3000}


def test_multiple_wallets_feed():

    exchange = mock.Mock()
    exchange.name = "fake"

    # Wallet 1
    w1 = Wallet(exchange, 10000*USD)

    lb1 = LockedBalance(w1)
    b1 = Balance(w1)

    w1 -= 1000 * USD
    w1 += Quantity(USD, 1000, path_id="fake_id_0")
    w1 -= 2000 * USD
    w1 += Quantity(USD, 2000, path_id="fake_id_1")

    tb1 = BinOp("fake.USD.total", operator.add)(b1, lb1)

    # Wallet 2
    w2 = Wallet(exchange, 10*BTC)

    lb2 = LockedBalance(w2)
    b2 = Balance(w2)

    w2 -= 3 * BTC
    w2 += Quantity(BTC, 3, path_id="fake_id_0")
    w2 -= 1 * BTC
    w2 += Quantity(BTC, 1, path_id="fake_id_1")

    tb2 = BinOp("fake.BTC.total", operator.add)(b2, lb2)

    feed = DataFeed([b1, lb1, tb1, b2, lb2, tb2])

    assert feed.next() == {"fake_USD": 7000, "fake_USD_locked": 3000, "fake.USD.total": 10000,
                           "fake_BTC": 6, "fake_BTC_locked": 4, "fake.BTC.total": 10}
