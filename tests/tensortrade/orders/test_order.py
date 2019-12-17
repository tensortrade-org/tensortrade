
import pytest
import pandas as pd
import ssl

from tensortrade.instruments import *
from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade.orders import Order, Broker
from tensortrade.orders.recipe import Recipe
from tensortrade.orders.criteria import StopLoss, StopDirection
from tensortrade.wallets import Portfolio, Wallet
from tensortrade.trades import TradeSide, TradeType

PRICE_COLUMN = "close"
data_frame = pd.read_csv("tests/data/input/coinbase-1h-btc-usd.csv")
data_frame.columns = map(str.lower, data_frame.columns)
data_frame = data_frame.rename(columns={'volume btc': 'volume'})

exchange = SimulatedExchange(
    data_frame=data_frame, price_column=PRICE_COLUMN, randomize_time_slices=True)
broker = Broker(exchange)
broker.reset()


def test_init():
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
