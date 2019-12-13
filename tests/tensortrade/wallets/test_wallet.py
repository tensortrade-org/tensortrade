

from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade.wallets import Wallet, Portfolio
from tensortrade.instruments import USD, BTC
from tensortrade.orders import Order
from tensortrade.trades import TradeSide, TradeType
from tensortrade.orders.criteria import HiddenLimit


exchange = SimulatedExchange()

wallet = Wallet(exchange, 10000*USD)

portfolio = Portfolio(base_instrument=USD, wallets=[wallet])

criteria = HiddenLimit(limit_price=10000)
order1 = Order(TradeSide.SELL, TradeType.LIMIT, USD/BTC, 765 *
               USD, portfolio=portfolio, criteria=criteria)

criteria = HiddenLimit(limit_price=12000)
order2 = Order(TradeSide.SELL, TradeType.MARKET, BTC/USD,
               334*USD, portfolio=portfolio, criteria=criteria)


def test_wallet_initialization():

    wallet = Wallet(exchange, 10000*USD)

    assert wallet.balance == 10000*USD
    assert wallet.exchange == exchange
    assert wallet.instrument == USD


def test_wallet_ops():
    pass
