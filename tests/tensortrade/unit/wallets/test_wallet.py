
import pytest
import unittest.mock as mock

from decimal import Decimal

from tensortrade.base.exceptions import InsufficientFunds, IncompatibleInstrumentOperation
from tensortrade.exchanges import Exchange
from tensortrade.wallets import Wallet
from tensortrade.instruments import USD, BTC, Quantity, ExchangePair


path_id = "f4cfeeae-a3e4-42e9-84b9-a24ccd2eebeb"
other_id = "7f3de243-0474-48d9-bf44-ca55ae07a70e"


exchange = Exchange('Exchange', lambda x: x)


def test_init():

    wallet = Wallet(exchange, 10000*USD)

    assert wallet.balance == 10000*USD
    assert wallet.exchange == exchange
    assert wallet.instrument == USD
    assert len(wallet.locked) == 0


def test_from_tuple():

    wallet_tuple = (exchange, USD, 10000)

    wallet = Wallet.from_tuple(wallet_tuple)

    assert wallet.balance == 10000*USD
    assert wallet.exchange == exchange
    assert wallet.instrument == USD
    assert len(wallet.locked) == 0


def test_valid_iadd():

    # Add to free unlocked balance
    wallet = Wallet(exchange, 10000 * USD)
    wallet += 500 * USD
    assert wallet.balance == 10500 * USD
    assert len(wallet.locked) == 0

    # Add to balance with locked path_id
    wallet = Wallet(exchange, 10000 * USD)
    wallet += Quantity(USD, 500, path_id=path_id)
    assert wallet.balance == 10000 * USD
    assert wallet.locked[path_id] == 500 * USD

    # Add to more balance with locked path_id
    wallet += Quantity(USD, 500, path_id=path_id)
    assert wallet.balance == 10000 * USD
    assert wallet.locked[path_id] == 1000 * USD

    # Add to balance that has another locked path_id
    wallet += Quantity(USD, 500, path_id=other_id)
    assert wallet.balance == 10000 * USD
    assert wallet.locked[path_id] == 1000 * USD
    assert wallet.locked[other_id] == 500 * USD


def test_invalid_iadd():

    # Add to free unlocked balance with different instrument
    wallet = Wallet(exchange, 10000 * USD)

    with pytest.raises(IncompatibleInstrumentOperation):
        wallet += 500 * BTC


def test_valid_isub():

    # Add to remove from unlocked balance
    wallet = Wallet(exchange, 10000 * USD)
    wallet -= 500 * USD
    assert wallet.balance == 9500 * USD
    assert len(wallet.locked) == 0

    # Add to balance with locked path_id
    wallet = Wallet(exchange, 10000 * USD)
    wallet += Quantity(USD, 750, path_id=path_id)
    wallet += Quantity(USD, 1000, path_id=other_id)

    wallet -= Quantity(USD, 500, path_id=path_id)
    assert wallet.balance == 10000 * USD
    assert wallet.locked[path_id] == 250 * USD

    wallet -= Quantity(USD, 500, path_id=other_id)
    assert wallet.balance == 10000 * USD
    assert wallet.locked[other_id] == 500 * USD


def test_invalid_isub():

    # Add to balance with locked path_id
    wallet = Wallet(exchange, 10000 * USD)
    wallet += Quantity(USD, 500, path_id=path_id)
    wallet += Quantity(USD, 700, path_id=other_id)

    with pytest.raises(InsufficientFunds):
        wallet -= 11000 * USD

    with pytest.raises(InsufficientFunds):
        wallet -= Quantity(USD, 750, path_id)

    with pytest.raises(InsufficientFunds):
        wallet -= Quantity(USD, 750, path_id)

    with pytest.raises(IncompatibleInstrumentOperation):
        wallet -= 500 * BTC


def test_deallocate():
    wallet = Wallet(exchange, 10000 * USD)
    wallet += Quantity(USD, 500, path_id=path_id)
    wallet += Quantity(USD, 700, path_id=other_id)

    assert wallet.balance == 10000 * USD
    assert wallet.locked[path_id] == 500 * USD
    assert wallet.locked[other_id] == 700 * USD

    wallet.deallocate(path_id)

    assert wallet.balance == 10500 * USD
    assert path_id not in wallet.locked.keys()
    assert wallet.locked[other_id] == 700 * USD

    wallet.deallocate(other_id)

    assert wallet.balance == 11200 * USD
    assert other_id not in wallet.locked.keys()


def test_locked_balance():

    wallet = Wallet(exchange, 10000 * USD)
    wallet += Quantity(USD, 500, path_id=path_id)
    wallet += Quantity(USD, 700, path_id=other_id)

    assert wallet.locked_balance == 1200 * USD


def test_total_balance():

    wallet = Wallet(exchange, 10000 * USD)
    wallet += Quantity(USD, 500, path_id=path_id)
    wallet += Quantity(USD, 700, path_id=other_id)

    assert wallet.total_balance == 11200 * USD


def test_transfer():

    exchange = mock.Mock()
    price = Decimal(9750.19).quantize(Decimal(10)**-2)
    exchange.quote_price = lambda pair: price
    exchange.name = "coinbase"

    order = mock.Mock()
    order.path_id = "fake_id"

    exchange_pair = ExchangePair(exchange, USD / BTC)

    source = Wallet(exchange, 18903.89 * USD)
    source.lock(917.07 * USD, order, "test")

    target = Wallet(exchange, 3.79283997 * BTC)

    quantity = (256.19 * USD).lock_for("fake_id")
    commission = (2.99 * USD).lock_for("fake_id")

    Wallet.transfer(source,
                    target,
                    quantity,
                    commission,
                    exchange_pair,
                    "transfer")

    source = Wallet(exchange, 3.79283997 * BTC)
    source.lock(3.00000029 * BTC, order, "test")

    target = Wallet(exchange, 18903.89 * USD)

    quantity = (2.19935873 * BTC).lock_for("fake_id")
    commission = (0.00659732 * BTC).lock_for("fake_id")

    Wallet.transfer(source,
                    target,
                    quantity,
                    commission,
                    exchange_pair,
                    "transfer")

