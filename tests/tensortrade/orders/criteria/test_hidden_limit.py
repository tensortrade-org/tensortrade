
import pytest

from tensortrade.base.clock import Clock
from tensortrade.orders.criteria import HiddenLimit
from tensortrade.instruments import USD, BTC
from tensortrade.trades.trade import TradeSide
from tensortrade.instruments import TradingPair


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def exchange(clock):

    class MockExchange:

        def __init__(self):
            self.base_instrument = USD
            self.id = "fake_id"
            self.clock = clock

        def quote_price(self, pair: 'TradingPair') -> float:
            d = {}
            if self.clock.step == 0:
                d = {
                    ("USD", "BTC"): 7117.00,
                }
            elif self.clock.step == 1:
                d = {
                    ("USD", "BTC"): 6750.00,
                }
            elif self.clock.step == 2:
                d = {
                    ("USD", "BTC"): 7000.00,
                }
            return d[(pair.base.symbol, pair.quote.symbol)]

        def is_pair_tradable(self, pair: TradingPair):
            if str(pair) == "USD/BTC":
                return True
            return False

    return MockExchange()


class MockOrder:

    def __init__(self, side, pair):
        self.side = side
        self.pair = pair


@pytest.fixture
def buy_order():
    return MockOrder(side=TradeSide.BUY, pair=USD/BTC)


@pytest.fixture
def sell_order():
    return MockOrder(side=TradeSide.SELL, pair=USD/BTC)


def test_init():
    criteria = HiddenLimit(limit_price=7000.00)
    assert criteria.limit_price == 7000.00


def test_call(buy_order, sell_order, exchange):

    criteria = HiddenLimit(limit_price=7000.00)

    assert not criteria(buy_order, exchange)
    assert criteria(sell_order, exchange)

    exchange.clock.increment()

    assert criteria(buy_order, exchange)
    assert not criteria(sell_order, exchange)

    exchange.clock.increment()

    assert criteria(buy_order, exchange)
    assert criteria(sell_order, exchange)


def test_str():
    criteria = HiddenLimit(limit_price=7000.00)

    assert str(criteria) == "<Limit: price={0}>".format(7000.00)
