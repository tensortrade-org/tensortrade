
import pytest
import pandas as pd

from tensortrade.base.exceptions import InsufficientFundsForAllocation, IncompatibleInstrumentOperation
from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade.wallets import Wallet
from tensortrade.instruments import USD, BTC, Quantity


path_id = "f4cfeeae-a3e4-42e9-84b9-a24ccd2eebeb"
other_id = "7f3de243-0474-48d9-bf44-ca55ae07a70e"

PRICE_COLUMN = "close"
data_frame = pd.read_csv("tests/data/input/coinbase-1h-btc-usd.csv")
data_frame.columns = map(str.lower, data_frame.columns)
data_frame = data_frame.rename(columns={'volume btc': 'volume'})

exchange = SimulatedExchange(data_frame=data_frame,
                             price_column=PRICE_COLUMN,
                             randomize_time_slices=True)


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

    with pytest.raises(InsufficientFundsForAllocation):
        wallet -= 11000 * USD

    with pytest.raises(InsufficientFundsForAllocation):
        wallet -= Quantity(USD, 750, path_id)

    with pytest.raises(InsufficientFundsForAllocation):
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
