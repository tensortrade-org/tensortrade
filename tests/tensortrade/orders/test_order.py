
import pytest

from tensortrade.base.clock import Clock
from tensortrade.instruments import *
from tensortrade.orders import Order, Broker
from tensortrade.orders.recipe import Recipe
from tensortrade.orders.criteria import StopLoss, StopDirection
from tensortrade.wallets import Portfolio, Wallet
from tensortrade.trades import TradeSide, TradeType


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
            if self.clock.step == 0:
                d = {
                    ("USD", "BTC"): 7117.00,
                    ("USD", "ETH"): 143.00,
                    ("USD", "XRP"): 0.22
                }
                return d[(pair.base.symbol, pair.quote.symbol)]
            d = {
                ("USD", "BTC"): 6750.00,
                ("USD", "ETH"): 135.00,
                ("USD", "XRP"): 0.30
            }
            return d[(pair.base.symbol, pair.quote.symbol)]

    return MockExchange()


@pytest.fixture
def wallet_usd(exchange):
    return Wallet(exchange, 10000 * USD)


@pytest.fixture
def wallet_btc(exchange):
    return Wallet(exchange, 1 * BTC)


@pytest.fixture
def wallet_eth(exchange):
    return Wallet(exchange, 10 * ETH)


@pytest.fixture
def wallet_xrp(exchange):
    return Wallet(exchange, 5000 * XRP)


@pytest.fixture
def portfolio(wallet_usd, wallet_btc, wallet_eth, wallet_xrp):

    portfolio = Portfolio(USD, wallets=[
        wallet_usd,
        wallet_btc,
        wallet_eth,
        wallet_xrp
    ])

    return portfolio


@pytest.fixture
def broker(exchange):
    return Broker(exchange)


def test_init(exchange, portfolio):

    base_wallet = portfolio.get_wallet(exchange.id, USD)

    quantity = (1 / 10) * base_wallet.balance
    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD/BTC,
                  quantity=quantity,
                  price=7200.00,
                  portfolio=portfolio)
    assert order
    assert order.id
    assert order.path_id
    assert order.quantity.instrument == USD
    assert order.filled_size == 0
    assert order.remaining_size == order.quantity
    assert isinstance(order.pair, TradingPair)
    assert order.pair.base == USD
    assert order.pair.quote == BTC


def test_adding_recipe():
    wallets = [
        Wallet(exchange, 10000 * USD),
        Wallet(exchange, 0 * BTC)
    ]
    portfolio = Portfolio(base_instrument=USD, wallets=wallets)
    base_wallet = portfolio.get_wallet(exchange.id, USD)

    quantity = (1 / 10) * base_wallet.balance
    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD/BTC,
                  quantity=quantity,
                  portfolio=portfolio)

    order += Recipe(
        side=TradeSide.SELL,
        trade_type=TradeType.MARKET,
        pair=BTC/USD,
        criteria=StopLoss(direction=StopDirection.EITHER, up_percent=0.02, down_percent=0.10),
    )
    assert order.pair


def test_buy_on_exchange():
    wallets = [
        Wallet(exchange, 10000 * USD),
        Wallet(exchange, 0 * BTC)
    ]
    portfolio = Portfolio(base_instrument=USD, wallets=wallets)

    base_wallet = portfolio.get_wallet(exchange.id, USD)
    quote_wallet = portfolio.get_wallet(exchange.id, BTC)

    quantity = (1 / 10) * base_wallet.balance
    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD/BTC,
                  quantity=quantity,
                  portfolio=portfolio)

    current_price = 12390.90

    base_wallet -= quantity.size * order.pair.base
    base_wallet += order.quantity

    trade = exchange._execute_buy_order(order, base_wallet, quote_wallet, current_price)

    assert trade
    assert trade.size == 1000 * USD
    assert trade.commission == 3 * USD


def test_sell_on_exchange():
    wallets = [
        Wallet(exchange, 0 * USD),
        Wallet(exchange, 10 * BTC)
    ]
    portfolio = Portfolio(base_instrument=USD, wallets=wallets)

    base_wallet = portfolio.get_wallet(exchange.id, USD)
    quote_wallet = portfolio.get_wallet(exchange.id, BTC)

    quantity = (1 / 10) * quote_wallet.balance
    order = Order(side=TradeSide.SELL,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=quantity,
                  portfolio=portfolio)

    current_price = 12390.90

    quote_wallet -= quantity.size * order.pair.quote
    quote_wallet += order.quantity

    trade = exchange._execute_sell_order(order, base_wallet, quote_wallet, current_price)

    assert trade
    assert trade.size == 0.997 * BTC
    assert trade.commission == 0.003 * BTC


def test_order_runs_through_broker():
    wallets = [
        Wallet(exchange, 10000 * USD),
        Wallet(exchange, 0 * BTC)
    ]
    portfolio = Portfolio(base_instrument=USD, wallets=wallets)
    exchange.reset()
    portfolio.reset()
    broker.reset()

    base_wallet = portfolio.get_wallet(exchange.id, USD)

    quantity = (1 / 10) * base_wallet.balance
    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=quantity,
                  portfolio=portfolio)

    base_wallet -= quantity.size * order.pair.base
    base_wallet += order.quantity

    broker.submit(order)

    broker.update()
    portfolio.update()


def test_path_order_runs_though_broker():

    wallets = [
        Wallet(exchange, 10000 * USD),
        Wallet(exchange, 0 * BTC)
    ]
    portfolio = Portfolio(base_instrument=USD, wallets=wallets)
    exchange.reset()
    portfolio.reset()
    broker.reset()

    base_wallet = portfolio.get_wallet(exchange.id, USD)

    quantity = (1 / 10) * base_wallet.balance
    order = Order(side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=quantity,
                  portfolio=portfolio)

    order = order.add_recipe(
        Recipe(
            side=TradeSide.SELL,
            trade_type=TradeType.MARKET,
            pair=USD/BTC,
            criteria=StopLoss(direction=StopDirection.EITHER, up_percent=0.02, down_percent=0.10)
        )
    )

    broker.submit(order)

    while len(broker.unexecuted) > 0:
        broker.update()
        portfolio.update()
        obs = exchange.next_observation(1)

    pytest.fail("Failed.")
