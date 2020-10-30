
import unittest.mock as mock

from tensortrade.oms.exchanges import ExchangeOptions
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.orders.create import proportion_order
from tensortrade.oms.wallets import Wallet, Portfolio


@mock.patch('tensortrade.exchanges.Exchange')
def test_proportion_order_init(mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.options = ExchangeOptions()
    exchange.id = "fake_id"
    exchange.name = "fake_exchange"

    wallet_usd = Wallet(exchange, 10000 * USD)
    wallet_btc = Wallet(exchange, 0 * BTC)
    portfolio = Portfolio(USD, [
        wallet_usd,
        wallet_btc
    ])

    order = proportion_order(
        portfolio=portfolio,
        source=wallet_usd,
        target=wallet_btc,
        proportion=1.0
    )

    assert order
