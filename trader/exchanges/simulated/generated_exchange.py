import numpy as np
import pandas as pd

from gym import spaces
from stochastic.continuous import FractionalBrownianMotion

from trader.environments import TradeType
from trader.exchanges.asset_exchange import AssetExchange


class GeneratedExchange(AssetExchange):
    def __init__(self, commission_percent: float = 0.3, base_precision: float = 1E-3, asset_precision: float = 1E-8, **kwargs):
        super(GeneratedExchange, self).__init__(commission_percent, base_precision, asset_precision)

        self.min_order_amount = kwargs.get('min_order_amount', 1E-5)
        self.base_price = kwargs.get('base_price', 3000)
        self.base_volume = kwargs.get('base_volume', 1000)
        self.hurst = kwargs.get('hurst', 0.68)

        self.fbm = FractionalBrownianMotion(t=1, hurst=self.hurst)

        self.reset()

    def reset(self):
        self.data_frame = pd.DataFrame([], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        self.data_frame.reset_index(drop=True)

    def observation_space(self):
        return spaces.Box(low=0, high=1, shape=(1, 5), dtype=self.dtype)

    def current_price(self):
        if len(self.data_frame) is 0:
            self.next_observation()

        return float(self.data_frame.loc[-1, 'close'])

    def _buy_price(self, current_price, commission, slippage):
        price_adjustment = price_adjustment = (1 + commission) * (1 + slippage)
        return round(current_price * price_adjustment, self.base_precision)

    def _sell_price(self, current_price, commission, slippage):
        price_adjustment = (1 - commission) * (1 - slippage)
        return round(current_price * price_adjustment, self.base_precision)

    def execute_trade(self, symbol: str, trade_type: TradeType, amount: float, price: float):
        current_price = self.current_price()

        commission = self.commission_percent / 100
        slippage = np.random.uniform(0, self.max_slippage_percent) / 100

        fill_price = 0

        balance = -1
        asset_held = -1

        if trade_type == TradeType.BUY and amount >= self.min_order_amount:
            fill_price = self._buy_price(current_price, commission, slippage)
            fill_amount = round((price * amount) / fill_price, self.asset_precision)
        elif trade_type == TradeType.SELL and amount >= self.min_order_amount:
            fill_price = self._sell_price(current_price, commission, slippage)
            fill_amount = round(amount, self.asset_precision)

        return fill_amount, fill_price

    def has_next_observation(self):
        return True

    def next_observation(self):
        dates = fbm.times(1)
        prices = fbm.sample(1)
        volumes = fbm.sample(1)

        price_frame = pd.DataFrame([], columns=['date', 'price'], dtype=float)
        volume_frame = pd.DataFrame([], columns=['date', 'volume'], dtype=float)

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

        self.data_frame.append({
            'open': float(ohlc['open'].values[-1] * self.base_price),
            'high': float(ohlc['high'].values[-1] * self.base_price),
            'low': float(ohlc['low'].values[-1] * self.base_price),
            'close': float(ohlc['close'].values[-1] * self.base_price),
            'volume': float(volume.values[-1] * self.base_volume),
        })

        df_min = self.data_frame.min()
        df_max = self.data_frame.max()

        normalized_df = (self.data_frame - df_min) / (df_max - df_min)

        return normalized_df.values[-1].astype(self.dtype)
