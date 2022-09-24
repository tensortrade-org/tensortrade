
import pytest
import unittest.mock as mock

from tensortrade.core.base import Clock
from tensortrade.oms.instruments.quantity import NegativeQuantity
from tensortrade.oms.wallets import Wallet, MarginWallet, Portfolio
from tensortrade.oms.instruments import USD, BTC, ETH, XRP, BCH


@pytest.fixture
@mock.patch('tensortrade.exchanges.Exchange')
def exchange(mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.base_instrument = USD
    exchange.id = "fake_id"

    exchange.clock = Clock()

    def quote_price(pair):
        if exchange.clock.step == 0:
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

    exchange.quote_price = mock.Mock(side_effect=quote_price)

    return exchange


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
def portfolio(wallet_usd, wallet_btc, wallet_eth, wallet_xrp, exchange):

    portfolio = Portfolio(USD, wallets=[
        wallet_usd,
        wallet_btc,
        wallet_eth,
        wallet_xrp
    ])

    with mock.patch.object(Portfolio, 'clock', return_value=exchange.clock) as clock:
        portfolio = Portfolio(USD, wallets=[
            wallet_usd,
            wallet_btc,
            wallet_eth,
            wallet_xrp
        ])

    return portfolio


@pytest.fixture
def portfolio_locked(portfolio, wallet_usd, wallet_btc, wallet_eth, wallet_xrp):
    def allocate(wallet, amount, identifier):
        wallet.withdraw(
            quantity=amount,
            reason="test"
        )
        wallet.deposit(
            quantity=amount.lock_for(identifier),
            reason="test"
        )
        return wallet

    wallet_usd = allocate(wallet_usd, 50 * USD, "1")
    wallet_usd = allocate(wallet_usd, 100 * USD, "2")
    wallet_btc = allocate(wallet_btc, 0.5 * BTC, "3")
    wallet_btc = allocate(wallet_btc, 0.25 * BTC, "4")
    wallet_eth = allocate(wallet_eth, 5 * ETH, "5")
    wallet_eth = allocate(wallet_eth, 2 * ETH, "6")
    wallet_xrp = allocate(wallet_xrp, 250 * XRP, "7")
    wallet_xrp = allocate(wallet_xrp, 12 * XRP, "8")

    return portfolio


def test_init_empty():

    portfolio = Portfolio(USD)

    assert portfolio.base_instrument == USD
    assert portfolio.base_balance == 0 * USD
    assert portfolio.initial_balance == 0 * USD
    assert len(portfolio.wallets) == 0


def test_init_from_wallets(exchange):

    portfolio = Portfolio(USD, wallets=[
        Wallet(exchange, 10000 * USD),
        Wallet(exchange, 0 * BTC)
    ])

    assert portfolio.base_instrument == USD
    assert portfolio.base_balance == 10000 * USD
    assert portfolio.initial_balance == 10000 * USD
    assert len(portfolio.wallets) == 2


def test_init_from_wallet_tuples(exchange):

    portfolio = Portfolio(USD, wallets=[
        (exchange, USD, 10000),
        (exchange, BTC, 0)
    ])

    assert portfolio.base_instrument == USD
    assert portfolio.base_balance == 10000 * USD
    assert portfolio.initial_balance == 10000 * USD
    assert len(portfolio.wallets) == 2


def test_balance(portfolio_locked):

    assert portfolio_locked.balance(USD) == 9850
    assert portfolio_locked.balance(BTC) == 0.25
    assert portfolio_locked.balance(ETH) == 3
    assert portfolio_locked.balance(XRP) == 4738


def test_locked_balance(portfolio_locked):

    assert portfolio_locked.locked_balance(USD) == 150
    assert portfolio_locked.locked_balance(BTC) == 0.75
    assert portfolio_locked.locked_balance(ETH) == 7
    assert portfolio_locked.locked_balance(XRP) == 262


def test_total_balance(portfolio_locked):

    assert portfolio_locked.total_balance(USD) == 10000
    assert portfolio_locked.total_balance(BTC) == 1
    assert portfolio_locked.total_balance(ETH) == 10
    assert portfolio_locked.total_balance(XRP) == 5000


def test_balances(portfolio_locked):

    balances = [9850 * USD, 0.25 * BTC, 3 * ETH, 4738 * XRP]
    assert portfolio_locked.balances == balances


def test_locked_balances(portfolio_locked):

    locked_balances = [150 * USD, 0.75 * BTC, 7 * ETH, 262 * XRP]
    assert portfolio_locked.locked_balances == locked_balances


def test_total_balances(portfolio_locked):

    total_balances = [10000 * USD, 1 * BTC, 10 * ETH, 5000 * XRP]
    assert portfolio_locked.total_balances == total_balances


def test_get_wallet(exchange, portfolio, wallet_xrp):

    assert wallet_xrp == portfolio.get_wallet(exchange.id, XRP)


def test_add(portfolio, exchange):

    wallet_bch = Wallet(exchange, 1000 * BCH)

    portfolio.add(wallet_bch)

    assert wallet_bch in portfolio.wallets


def test_remove(portfolio, wallet_btc):

    portfolio.remove(wallet_btc)

    assert wallet_btc not in portfolio.wallets


def test_remove_pair(portfolio, exchange):

    portfolio.remove_pair(exchange, BTC)

    assert wallet_btc not in portfolio.wallets

def test_init_with_short_positions(exchange):
    
    portfolio = Portfolio(USD, wallets=[
        Wallet(exchange, 1000 * USD),
        MarginWallet(exchange, NegativeQuantity(BTC, -1))
    ])
    assert portfolio.base_instrument == USD
    assert portfolio.base_balance == 1000 * USD
    assert portfolio.initial_balance == 1000 * USD
    assert len(portfolio.wallets) == 2
