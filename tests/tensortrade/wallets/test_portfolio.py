
import pytest
import pandas as pd

from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade.wallets import Wallet, Portfolio
from tensortrade.instruments import USD, BTC, ETH, XRP


PRICE_COLUMN = "close"
data_frame = pd.read_csv("tests/data/input/coinbase-1h-btc-usd.csv")
data_frame.columns = map(str.lower, data_frame.columns)
data_frame = data_frame.rename(columns={'volume btc': 'volume'})

coinbase = SimulatedExchange(data_frame=data_frame,
                             price_column=PRICE_COLUMN,
                             randomize_time_slices=True)


wallets = [
    Wallet(coinbase, 10000 * USD),
    Wallet(coinbase, 0 * BTC)
]

wallet_tuples = [
   (coinbase, USD, 10000),
   (coinbase, BTC, 0)
]


def test_init():

    # Empty portfolio
    portfolio = Portfolio(USD)

    assert portfolio.base_instrument == USD
    assert portfolio.base_balance == 0 * USD
    assert portfolio.initial_balance == 0 * USD
    assert len(portfolio.wallets) == 0

    # Coinbase 2-wallet portfolio from List[Wallet]
    portfolio = Portfolio(USD, wallets=wallets)

    assert portfolio.base_instrument == USD
    assert portfolio.base_balance == 10000 * USD
    assert portfolio.initial_balance == 10000 * USD
    assert len(portfolio.wallets) == 2

    # Coinbase 2-wallet portfolio from List[Tuple['Exchange', Instrument, float]]
    portfolio = Portfolio(USD, wallets=wallet_tuples)

    assert portfolio.base_instrument == USD
    assert portfolio.base_balance == 10000 * USD
    assert portfolio.initial_balance == 10000 * USD
    assert len(portfolio.wallets) == 2


def test_net_worth():

    wallet_usd = Wallet(coinbase, 10000 * USD)
    wallet_btc = Wallet(coinbase, 1 * BTC)
    wallet_eth = Wallet(coinbase, 10 * ETH)
    wallet_xrp = Wallet(coinbase, 5000 * XRP)

    def allocate(wallet, amount, identifier):
        wallet -= amount
        amount.lock_for(identifier)
        wallet += amount
        return wallet

    wallet_usd = allocate(wallet_usd, 50 * USD, "1")
    wallet_usd = allocate(wallet_usd, 100 * USD, "2")
    wallet_btc = allocate(wallet_btc, 0.5 * BTC, "3")
    wallet_btc = allocate(wallet_btc, 0.25 * BTC, "4")
    wallet_eth = allocate(wallet_eth, 5 * ETH, "5")
    wallet_eth = allocate(wallet_eth, 2 * ETH, "6")
    wallet_xrp = allocate(wallet_xrp, 250 * XRP, "7")
    wallet_xrp = allocate(wallet_xrp, 2.5 * XRP, "8")

    pytest.fail("Failed.")


def test_profit_loss():
    pytest.fail("Failed.")


def test_performance():
    pytest.fail("Failed.")


def test_balance():
    wallet_usd = Wallet(coinbase, 10000 * USD)
    wallet_btc = Wallet(coinbase, 1 * BTC)
    wallet_eth = Wallet(coinbase, 10 * ETH)
    wallet_xrp = Wallet(coinbase, 5000 * XRP)

    def allocate(wallet, amount, identifier):
        wallet -= amount
        amount.lock_for(identifier)
        wallet += amount
        return wallet

    wallet_usd = allocate(wallet_usd, 50 * USD, "1")
    wallet_usd = allocate(wallet_usd, 100 * USD, "2")
    wallet_btc = allocate(wallet_btc, 0.5 * BTC, "3")
    wallet_btc = allocate(wallet_btc, 0.25 * BTC, "4")
    wallet_eth = allocate(wallet_eth, 5 * ETH, "5")
    wallet_eth = allocate(wallet_eth, 2 * ETH, "6")
    wallet_xrp = allocate(wallet_xrp, 250 * XRP, "7")
    wallet_xrp = allocate(wallet_xrp, 30 * XRP, "8")

    portfolio = Portfolio(USD, wallets=[wallet_usd, wallet_btc, wallet_eth, wallet_xrp])

    assert portfolio.balance(USD) == 9850
    assert portfolio.balance(BTC) == 0.25
    assert portfolio.balance(ETH) == 3
    assert portfolio.balance(XRP) == 4720


def test_locked_balance():
    pytest.fail("Failed.")


def test_total_balance():
    pytest.fail("Failed.")


def test_get_wallet():
    pytest.fail("Failed.")


def test_add():
    wallet_usd = Wallet(coinbase, 10000 * USD)
    wallet_btc = Wallet(coinbase, 1 * BTC)
    wallet_eth = Wallet(coinbase, 10 * ETH)
    wallet_xrp = Wallet(coinbase, 5000 * XRP)
    portfolio = Portfolio(USD, wallets=[
        wallet_usd,
        wallet_btc,
        wallet_eth
    ])

    portfolio.add(wallet_xrp)

    assert wallet_xrp in portfolio.wallets


def test_remove():
    wallet_usd = Wallet(coinbase, 10000 * USD)
    wallet_btc = Wallet(coinbase, 1 * BTC)
    wallet_eth = Wallet(coinbase, 10 * ETH)
    wallet_xrp = Wallet(coinbase, 5000 * XRP)
    portfolio = Portfolio(USD, wallets=[
        wallet_usd,
        wallet_btc,
        wallet_eth,
        wallet_xrp
    ])

    portfolio.remove(wallet_btc)

    assert wallet_btc not in portfolio.wallets

    pytest.fail("Test removing base wallet and make sure there is an error.")


def test_remove_pair():
    wallet_usd = Wallet(coinbase, 10000 * USD)
    wallet_btc = Wallet(coinbase, 1 * BTC)
    wallet_eth = Wallet(coinbase, 10 * ETH)
    wallet_xrp = Wallet(coinbase, 5000 * XRP)
    portfolio = Portfolio(USD, wallets=[
        wallet_usd,
        wallet_btc,
        wallet_eth,
        wallet_xrp
    ])

    portfolio.remove_pair(coinbase, BTC)

    assert wallet_btc not in portfolio.wallets

    pytest.fail("Test removing base wallet and make sure there is an error.")


def test_update():
    pytest.fail("Failed.")


def test_reset():
    pytest.fail("Failed.")
