

from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade.wallets import Wallet
from tensortrade.instruments import *
from tensortrade.orders import Order
from tensortrade.trades import TradeSide, TradeType
from tensortrade.orders.criteria import Limit


exchange = SimulatedExchange()


criteria = Limit(limit_price=10000)
order1 = Order(TradeSide.SELL, TradeType.LIMIT, USD/BTC, 765*USD, criteria=criteria)

criteria = Limit(limit_price=12000)
order2 = Order(BTC/USD, 334*USD, criteria)


def test_wallet_initialization():

    wallet = Wallet(exchange, 10000*USD)

    assert wallet.initial_balance == 10000*USD
    assert wallet.exchange == exchange
    assert wallet.instrument == USD


def test_wallet_ops():
    pass
