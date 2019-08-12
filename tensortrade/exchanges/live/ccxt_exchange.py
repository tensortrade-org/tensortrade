import time
import numpy as np
import pandas as pd

from gym import spaces
from typing import Dict
from ccxt import Exchange
from sklearn.preprocessing import MinMaxScaler

from tensortrade.environments.actions import TradeType
from tensortrade.exchanges.asset_exchange import AssetExchange


class CCXTExchange(AssetExchange):
    def __init__(self, exchange: Exchange, **kwargs):
        self.exchange = exchange

        self._markets = self.exchange.load_markets()

        self._initial_balance = self.exchange.fetch_free_balance()

        self.observation_type = kwargs.get('observation_type', 'trades')
        self.observation_symbol = kwargs.get('observation_symbol', 'ETH/BTC')
        self.observation_timeframe = kwargs.get('observation_timeframe', '10m')
        self.observation_window_size = kwargs.get('observation_window_size', 10)

        self.async_timeout_in_ms = kwargs.get('async_timeout_in_ms', 15)
        self.max_trade_wait_in_sec = kwargs.get('max_trade_wait_in_sec', 60)

        self._performance = pd.DataFrame([], columns=['balance', 'net_worth'])

    def net_worth(self, output_symbol: str = 'BTC') -> float:
        return super().net_worth(output_symbol=output_symbol)

    def profit_loss_percent(self, output_symbol: str = 'BTC') -> float:
        return super().profit_loss_percent(output_symbol=output_symbol)

    def initial_balance(self, symbol: str = 'BTC') -> float:
        return self._initial_balance

    def balance(self, symbol: str = 'BTC') -> float:
        return self.exchange.fetch_free_balance()[symbol]

    def portfolio(self) -> Dict[str, float]:
        portfolio = self.exchange.fetch_free_balance()

        return {k: v for k, v in portfolio.items() if v > 0}

    def trades(self) -> pd.DataFrame:
        trades = {}

        for key in self._markets.keys():
            trades[key] = self.exchange.fetch_my_trades()[key]

        return trades

    def performance(self) -> pd.DataFrame:
        return self._performance

    def observation_space(self):
        if self.observation_type == 'ohlcv':
            return spaces.Box(low=0, high=1, shape=(self.observation_window_size, 5), dtype=self.dtype)

        return spaces.Box(low=0, high=1, shape=(self.observation_window_size, 4), dtype=self.dtype)

    def current_price(self, symbol: str, output_symbol: str = 'BTC'):
        return self.exchange.fetch_ticker(symbol)['close']

    def execute_trade(self, symbol: str, trade_type: TradeType, amount: float, price: float):
        if trade_type == TradeType.BUY:
            order = self.exchange.create_order(symbol=symbol, type="limit",
                                               side="buy", amount=amount, price=price)
        elif trade_type == TradeType.BUY:
            order = self.exchange.create_order(symbol=symbol, type="limit",
                                               side="sell", amount=amount, price=price)

        max_wait_time = time.time() + self.max_trade_wait_in_sec

        while order['status'] is 'open' and time.time() < max_wait_time:
            time.sleep(self.exchange.rateLimit / 1000)

            order = self.exchange.fetch_order(order.id)

        if order['status'] is 'open':
            self.exchange.cancel_order(order.id)

        if order['filled'] > 0:
            self._performance.append({
                'balance': self.balance(),
                'net_worth': self.net_worth(),
            }, ignore_index=True)

    def has_next_observation(self):
        if self.observation_type == 'ohlcv':
            return self.exchange.has['fetchOHLCV']

        return self.exchange.has['fetchTrades']

    def next_observation(self):
        time.sleep(self.exchange.rateLimit / 1000)

        if self.observation_type == 'ohlcv':
            ohlcv = self.exchange.fetch_ohlcv(self.observation_symbol, self.observation_timeframe)

            obs = [l[1:] for l in ohlcv]

            if len(obs) < self.observation_window_size:
                return np.pad(obs, (self.observation_window_size - len(obs), 5))
        elif self.observation_type == 'trades':
            trades = self.exchange.fetch_trades(self.observation_symbol)

            obs = [[0 if t['side'] == 'buy' else 1, t['price'], t['amount'], t['cost']]
                   for t in trades]

            if len(obs) < self.observation_window_size:
                return np.pad(obs, (self.observation_window_size - len(obs), 4))

        scaler = MinMaxScaler()
        obs = scaler.fit_transform(obs)

        return obs
