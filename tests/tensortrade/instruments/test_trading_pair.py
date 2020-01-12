

import pytest

from tensortrade.instruments import Instrument, TradingPair, Price, BTC, USD
from tensortrade.base.exceptions import InvalidTradingPair, IncompatibleTradingPairOperation


def test_valid_init():

    pair = TradingPair(USD,  BTC)
    assert pair
    assert pair.base == USD
    assert pair.quote == BTC


def test_invalid_init():

    with pytest.raises(InvalidTradingPair):
        pair = TradingPair(BTC, BTC)


def test_valid_rmul():
    pair = TradingPair(USD, BTC)

    # int
    price = 8 * pair
    assert isinstance(price, Price)
    assert price.rate == 8
    assert price.pair == pair

    # float
    price = 8.0 * pair
    assert isinstance(price, Price)
    assert price.rate == 8
    assert price.pair == pair


def test_invalid_rmul():

    pair = TradingPair(USD, BTC)
    with pytest.raises(IncompatibleTradingPairOperation):
        "btc" * pair


def test_str():
    pair = TradingPair(USD, BTC)
    assert str(pair) == "USD/BTC"
