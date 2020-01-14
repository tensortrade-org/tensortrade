
import pytest
import unittest.mock as mock

from tensortrade.instruments import USD, BTC, ETH, Quantity
from tensortrade.wallets import Wallet, Portfolio
from tensortrade.data import PortfolioDataSource


@pytest.fixture
def portfolio():
    exchange = mock.Mock()

    exchange.id = "fake_id"

    portfolio = Portfolio(USD, [
        Wallet(exchange, 10000*USD),
        Wallet(exchange, 10 * BTC),
        Wallet(exchange, 200 * ETH)
    ])
    return portfolio


def tests_portfolio_data_source(portfolio):

    source = PortfolioDataSource(portfolio)
    assert source
    assert source.next() == {'USD': 10000.00, 'USD_pending': 0.00,
                             'BTC': 10.00000000, 'BTC_pending': 0.00000000,
                             'ETH': 200.00000000, 'ETH_pending': 0.00000000}

    wallet_usd = portfolio.get_wallet("fake_id", USD)

    wallet_usd -= 1000*USD
    wallet_usd += Quantity(USD, 1000, path_id="fake_path_id")

    assert source.next() == {'USD': 9000.00, 'USD_pending': 1000.00,
                             'BTC': 10.00000000, 'BTC_pending': 0.00000000,
                             'ETH': 200.00000000, 'ETH_pending': 0.00000000}
