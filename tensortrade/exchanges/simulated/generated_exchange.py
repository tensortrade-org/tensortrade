import numpy as np
import pandas as pd

from gym import spaces
from typing import Dict
from sklearn.preprocessing import MinMaxScaler
from stochastic.continuous import FractionalBrownianMotion

from tensortrade.environments.actions import TradeType
from tensortrade.exchanges.asset_exchange import AssetExchange


class GeneratedExchange(AssetExchange):
    def __init__(self, **kwargs):
        self.commission_percent = kwargs.get('commission_percent', 0.3)
        self.base_precision = kwargs.get('base_precision', 2)
        self.asset_precision = kwargs.get('asset_precision', 8)
        self._initial_balance = kwargs.get('initial_balance', 1E5)
        self.max_allowed_slippage_percent = kwargs.get(
            'max_allowed_slippage_percent', 3.0)
        self.min_order_amount = kwargs.get('min_order_amount', 1E-5)
        self.base_price = kwargs.get('base_price', 3000)
        self.base_volume = kwargs.get('base_volume', 1000)
        self.hurst = kwargs.get('hurst', 0.58)

        self.fbm = FractionalBrownianMotion(t=1, hurst=self.hurst)

        self.reset()

    def reset(self):
        self._balance = self._initial_balance

        self._portfolio = {}
        self._trades = pd.DataFrame(
            [], columns=['step', 'symbol', 'type', 'amount', 'price'])
        self._performance = pd.DataFrame([], columns=['balance', 'net_worth'])

        self.data_frame = pd.DataFrame(
            [], columns=['open', 'high', 'low', 'close', 'volume'])
        self.data_frame.reset_index(drop=True)

        self.current_step = 1

    def net_worth(self, output_symbol: str = 'USD') -> float:
        return super().net_worth(output_symbol=output_symbol)

    def profit_loss_percent(self, output_symbol: str = 'USD') -> float:
        return super().profit_loss_percent(output_symbol=output_symbol)

    def initial_balance(self, symbol: str = 'USD') -> float:
        return self._initial_balance

    def balance(self, symbol: str = 'USD') -> float:
        return self._balance

    def portfolio(self) -> Dict[str, float]:
        return self._portfolio

    def trades(self) -> pd.DataFrame:
        return self._trades

    def performance(self) -> pd.DataFrame:
        return self._performance

    def observation_space(self):
        return spaces.Box(low=0, high=1, shape=(1, 5), dtype=self.dtype)

    def current_price(self, symbol: str, output_symbol: str = 'USD'):
        if len(self.data_frame) is 0:
            self.next_observation()

        return float(self.data_frame['close'].values[-1])

    def _update_account(self, symbol: str, trade_type: TradeType, fill_amount: float, fill_price: float):
        self._trades = self._trades.append({
            'step': self.current_step,
            'symbol': symbol,
            'type': trade_type,
            'amount': fill_amount,
            'price': fill_price
        }, ignore_index=True)

        if trade_type is TradeType.BUY:
            self._balance -= fill_amount * fill_price
            self._portfolio[symbol] = self._portfolio.get(
                symbol, 0) + fill_amount
        elif trade_type is TradeType.SELL:
            self._balance += fill_amount * fill_price
            self._portfolio[symbol] -= fill_amount

        self._performance.append({
            'balance': self._balance,
            'net_worth': self.net_worth(),
        }, ignore_index=True)

    def execute_trade(self, symbol: str, trade_type: TradeType, amount: float, price: float):
        current_price = self.current_price(symbol=symbol)

        commission = self.commission_percent / 100
        slippage = np.random.uniform(
            0, self.max_allowed_slippage_percent) / 100

        fill_amount = 0

        if trade_type == TradeType.BUY and amount >= self.min_order_amount and self._balance >= amount * price:
            price_adjustment = price_adjustment = (
                1 + commission) * (1 + slippage)
            fill_price = round(
                current_price * price_adjustment, self.base_precision)
            fill_amount = round((price * amount) /
                                fill_price, self.asset_precision)
        elif trade_type == TradeType.SELL and amount >= self.min_order_amount and self._portfolio.get(symbol, 0) >= amount:
            price_adjustment = (1 - commission) * (1 - slippage)
            fill_price = round(
                current_price * price_adjustment, self.base_precision)
            fill_amount = round(amount, self.asset_precision)

        if fill_amount > 0:
            self._update_account(symbol=symbol,
                                 trade_type=trade_type,
                                 fill_amount=fill_amount,
                                 fill_price=fill_price)

    def has_next_observation(self):
        return True

    def next_observation(self):
        dates = np.linspace(0, 1, 2)
        prices = self.fbm.sample(1)
        volumes = self.fbm.sample(1)

        price_frame = pd.DataFrame([], columns=['date', 'price'], dtype=float)
        volume_frame = pd.DataFrame(
            [], columns=['date', 'volume'], dtype=float)

        price_frame['date'] = pd.to_datetime(dates, unit="m")
        price_frame['price'] = prices

        volume_frame['date'] = pd.to_datetime(dates, unit="m")
        volume_frame['volume'] = volumes

        price_frame.set_index('date')
        price_frame.index = pd.to_datetime(price_frame.index, unit='s')

        volume_frame.set_index('date')
        volume_frame.index = pd.to_datetime(price_frame.index, unit='s')

        ohlc = price_frame['price'].resample('1min').ohlc()
        volume = volume_frame['volume'].resample('1min').sum()

        ohlc.reset_index(drop=True)
        volume.reset_index(drop=True)

        self.data_frame = self.data_frame.append({
            'open': float(ohlc['open'].values[-1] * self.base_price),
            'high': float(ohlc['high'].values[-1] * self.base_price),
            'low': float(ohlc['low'].values[-1] * self.base_price),
            'close': float(ohlc['close'].values[-1] * self.base_price),
            'volume': float(volume.values[-1] * self.base_volume),
        }, ignore_index=True)

        self.current_step += 1

        scaler = MinMaxScaler()
        scaled_frame = scaler.fit_transform(self.data_frame.values)

        return scaled_frame[-1].astype(self.dtype)
