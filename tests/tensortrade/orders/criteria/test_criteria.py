
import pytest

from tensortrade.orders.criteria import Criteria, Limit, Timed, Stop
from tensortrade.base.clock import Clock
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


class ConcreteCriteria(Criteria):

    def call(self, order: 'Order', exchange: 'Exchange'):
        return True


def test_init():
    criteria = ConcreteCriteria()
    assert criteria


def test_call(exchange):

    criteria = ConcreteCriteria()

    order = MockOrder(side=TradeSide.BUY, pair=BTC/USD)
    assert not criteria(order, exchange)

    order = MockOrder(side=TradeSide.BUY, pair=USD/BTC)
    assert criteria(order, exchange)


def test_and():

    criteria = Limit(limit_price=7000.00) & Timed(wait=2)
    assert criteria
    assert isinstance(criteria, Criteria)

    order = MockOrder(side=TradeSide.BUY, pair=USD/BTC)


def test_or():

    criteria = Limit(limit_price=7000.00) | Timed(wait=2)
    assert criteria
    assert isinstance(criteria, Criteria)

    order = MockOrder(side=TradeSide.BUY, pair=USD/BTC)


def test_invert():
    pytest.fail("Failed.")