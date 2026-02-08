import pytest

from tensortrade.core.exceptions import InvalidTradingPair
from tensortrade.oms.instruments import BTC, USD, TradingPair


def test_valid_init():
    pair = TradingPair(USD, BTC)
    assert pair
    assert pair.base == USD
    assert pair.quote == BTC


def test_invalid_init():
    with pytest.raises(InvalidTradingPair):
        TradingPair(BTC, BTC)


def test_str():
    pair = TradingPair(USD, BTC)
    assert str(pair) == "USD/BTC"
