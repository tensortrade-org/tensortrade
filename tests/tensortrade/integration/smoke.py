
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


from tensortrade.utils import CryptoDataDownload
from tensortrade.wallets import Portfolio, Wallet
from tensortrade.exchanges import Exchange
from tensortrade.exchanges.services.execution.simulated import execute_order
from tensortrade.data import Stream
from tensortrade.instruments import USD, BTC, ETH, LTC
from tensortrade.environments import TradingEnvironment


cdd = CryptoDataDownload()
coinbase_btc = cdd.fetch("Coinbase", "USD", "BTC", "1h")
coinbase_eth = cdd.fetch("Coinbase", "USD", "ETH", "1h")

bitstamp_btc = cdd.fetch("Bitstamp", "USD", "BTC", "1h")
bitstamp_eth = cdd.fetch("Bitstamp", "USD", "ETH", "1h")
bitstamp_ltc = cdd.fetch("Bitstamp", "USD", "LTC", "1h")

ex1 = Exchange("coinbase", service=execute_order)(
    Stream("USD-BTC", list(coinbase_btc['close'][-100:])),
    Stream("USD-ETH", list(coinbase_eth['close'][-100:]))
)

ex2 = Exchange("binance", service=execute_order)(
    Stream("USD-BTC", list(bitstamp_btc['close'][-100:])),
    Stream("USD-ETH", list(bitstamp_eth['close'][-100:])),
    Stream("USD-LTC", list(bitstamp_ltc['close'][-100:]))
)

portfolio = Portfolio(USD, [
    Wallet(ex1, 10000 * USD),
    Wallet(ex1, 10 * BTC),
    Wallet(ex1, 5 * ETH),
    Wallet(ex2, 1000 * USD),
    Wallet(ex2, 5 * BTC),
    Wallet(ex2, 20 * ETH),
    Wallet(ex2, 3 * LTC)
])

env = TradingEnvironment(
    action_scheme="simple",
    reward_scheme="simple",
    portfolio=portfolio
)

done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

df = portfolio.ledger.as_frame()
df.to_clipboard(index=False)

