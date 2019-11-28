# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import numpy as np
import pandas as pd

import tensortrade.slippage as slippage

from gym.spaces import Space, Box
from typing import List, Dict

from tensortrade.trades import Trade, TradeType
from tensortrade.exchanges import Exchange
from tensortrade.features import FeaturePipeline
from tensortrade.slippage import RandomUniformSlippageModel

class SimulatedExchange(Exchange):
    """An exchange, in which the price history is based off the supplied data frame and
    trade execution is largely decided by the designated slippage model.

    If the `data_frame` parameter is not supplied upon initialization, it must be set before
    the exchange can be used within a trading environments.
    """

    def __init__(self, data_frame: pd.DataFrame = None, **kwargs):
        super().__init__(base_instrument=kwargs.get('base_instrument', 'USD'),
                         dtype=kwargs.get('dtype', np.float16),
                         feature_pipeline=kwargs.get('feature_pipeline', None))


        self._initial_balance = kwargs.get('initial_balance', 1E4)
        self._observation_columns = kwargs.get(
            'observation_columns', ['open', 'high', 'low', 'close', 'volume'])


        # critical for maintining the actual price after transformation
        self._close_column = kwargs.get('price_column', 'close')
        # critical for mainting the high and low of the candle after transformation.
        # used within the slip model to ensure slip bounds
        self._open_column = kwargs.get('open_column', 'open')
        self._high_column = kwargs.get('high_column', 'high')
        self._low_column = kwargs.get('low_column', 'low')

        self._window_size = kwargs.get('window_size', 1)
        self._pretransform = kwargs.get('pretransform', True)

        # dataframe for maintining pre-transformation price history
        self._price_history = None

        # Note: This is invoking the data_frame setter
        self.data_frame = data_frame

        self._max_price_slippage_percent = kwargs.get('max_allowed_slippage_percent', kwargs.get('_max_price_slippage_percent', 1.0))
        self._max_amount_slippage_percent = kwargs.get('max_allowed_slippage_percent', kwargs.get('_max_amount_slippage_percent', 0.0))

        SlippageModelClass = kwargs.get('slippage_model', RandomUniformSlippageModel)
        self._price_slip_model = SlippageModelClass(min=self._min_trade_price,
                                                    max=self._max_trade_price,
                                                    slip=self._max_price_slippage_percent)
        self._amount_slip_model = SlippageModelClass(min=self._min_trade_amount,
                                                    max=self._max_trade_amount,
                                                    slip=self._max_amount_slippage_percent)

    @property
    def window_size(self) -> int:
        """The window size of observations."""
        return self._window_size

    @window_size.setter
    def window_size(self, window_size: int):
        self._window_size = window_size

        if isinstance(self.data_frame, pd.DataFrame) and self._pretransform:
            self.transform_data_frame()

    @property
    def data_frame(self) -> pd.DataFrame:
        """The underlying data model backing the price and volume simulation."""
        return getattr(self, '_data_frame', None)

    @data_frame.setter
    def data_frame(self, data_frame: pd.DataFrame):
        if not isinstance(data_frame, pd.DataFrame):
            self._data_frame = data_frame
            self._price_history = None
            return

        self._data_frame = data_frame
        self._price_history = data_frame[[self._open_column, self._close_column, self._high_column, self._low_column]]

        if self._pretransform:
            self.transform_data_frame()

    @property
    def feature_pipeline(self) -> FeaturePipeline:
        return self._feature_pipeline

    @feature_pipeline.setter
    def feature_pipeline(self, feature_pipeline=FeaturePipeline):
        self._feature_pipeline = feature_pipeline

        if isinstance(self.data_frame, pd.DataFrame) and self._pretransform:
            self.transform_data_frame()

        return self._feature_pipeline

    @property
    def initial_balance(self) -> float:
        return self._initial_balance

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def portfolio(self) -> Dict[str, float]:
        return self._portfolio

    @property
    def trades(self) -> pd.DataFrame:
        return self._trades

    @property
    def performance(self) -> pd.DataFrame:
        return self._performance

    @property
    def observation_columns(self) -> List[str]:
        if self._data_frame is None:
            return None

        data_frame = self._data_frame.iloc[0:1]

        if self._feature_pipeline is not None:
            data_frame = self._feature_pipeline.transform(data_frame)

        return data_frame.select_dtypes(include=[np.float, np.number]).columns

    @property
    def has_next_observation(self) -> bool:
        return self._current_step < len(self._data_frame) - 1

    def _next_observation(self) -> pd.DataFrame:
        lower_range = max(self._current_step - self._window_size, 0)
        upper_range = min(self._current_step + 1, len(self._data_frame))

        obs = self._data_frame.iloc[lower_range:upper_range]

        if not self._pretransform and self._feature_pipeline is not None:
            obs = self._feature_pipeline.transform(obs)

        if len(obs) < self._window_size:
            padding = np.zeros((self._window_size - len(obs), len(self.observation_columns)))
            padding = pd.DataFrame(padding, columns=self.observation_columns)
            obs = pd.concat([padding, obs], ignore_index=True, sort=False)

        obs = obs.select_dtypes(include='number')

        self._current_step += 1

        return obs

    def transform_data_frame(self) -> bool:
        if self._feature_pipeline is not None:
            self._data_frame = self._feature_pipeline.transform(self._data_frame)


    def current_price(self, symbol: str) -> float:
        if self._price_history is not None:
            return float(self._price_history.iloc[self._current_step][self._close_column])
        # return np.inf

    def _is_valid_trade(self, trade: Trade) -> bool:
        if trade.valid:
            return True

        if trade.is_hold:
            return trade.transact_amount <= self.portfolio.get(trade.symbol, 0)
        elif trade.is_buy:
            return trade.transact_amount >= self._min_trade_amount and self.balance >= trade.transact_total
        elif trade.is_sell:
            return trade.transact_amount >= self._min_trade_amount and self.portfolio.get(trade.symbol, 0) >= trade.transact_amount

        return False

    def _make_trade(self, trade: Trade):
        if not trade.is_hold:
            self._trades = self._trades.append({
                'step': trade.step,
                'symbol': trade.symbol,
                'type': trade.trade_type,
                'amount': trade.amount,
                'price': trade.price
            }, ignore_index=True)

    def _update_account(self, trade: Trade):

        log = {'step': self._current_step}
        if trade.is_buy:
            log['action'] = trade.log
            self._balance -= trade.transact_total
            self._portfolio[trade.symbol] = self._portfolio.get(trade.symbol, 0) + trade.transact_amount
        elif trade.is_sell:
            log['action'] = trade.log
            self._balance += trade.transact_total - trade.transact_commission
            self._portfolio[trade.symbol] = self._portfolio.get(trade.symbol, 0) - trade.transact_amount
        elif trade.is_hold:
            log['action'] = trade.log
        else:
            log['action'] = "Unknown Trade Type: {}".format(trade.to_dict)

        if self._is_valid_trade(trade):
            self._make_trade(trade)
        if self._is_valid_trade(trade):
            self._trades = self._trades.append({'index': self.data_frame.iloc[self._current_step].name,
                                                'step': self._current_step,
                                                'symbol': trade.symbol,
                                                'type': trade.trade_type,
                                                'amount': trade.transact_amount,
                                                'price': trade.transact_total}, ignore_index=True)
        log.update({'balance': self.balance,
                    'net_worth': self.net_worth})

        self._performance = self._performance.append( log , ignore_index=True)

    def _update_account(self, trade: Trade):

        self._portfolio[self._base_instrument] = self.balance


    def execute_trade(self, trade: Trade) -> Trade:
        trade = trade.copy()
        transact_amount = trade.order_amount
        transact_price = trade.order_price

        if trade.is_hold:
            trade.transact_amount = transact_amount
            trade.transact_price = transact_price
        else:
            trade.order_commission = transact_price * self.commission_percent

        if trade.is_buy:
            if trade.is_limit_buy:
                # our buying power will fluxuate slightly as the market moves
                slip_adjusted_amount = (self.balance * (1-(self._max_amount_slippage_percent/100)) * transact_amount) / transact_price
                transact_amount = self._amount_slip_model.random_slip(slip_adjusted_amount)
            elif trade.is_market_buy:
                # the price can fluxuate up or down during the market buy order
                transact_price = self._price_slip_model.random_slip(transact_price)
            else:
                # catch all
                transact_price = self._bind_trade_price(transact_price)
                transact_amount = self._bind_trade_amount(transact_amount)

        elif trade.is_sell:
            if trade.is_limit_sell:
                # if we're selling we can only sell LESS than we have on hand.
                amount = self._amount_slip_model.slip_down(transact_amount)
            elif trade.is_market_sell:
                # sell price can fluxuate up or down during a market sale order
                transact_price = self._price_slip_model.random_slip(transact_price)
            else:
                # catch all
                transact_price = self._bind_trade_price(transact_price)
                transact_amount = self._bind_trade_amount(transact_amount)


        # save all the
        trade.transact_price = round(transact_price, self._base_precision)
        trade.transact_amount = round(transact_amount, self._instrument_precision)
        trade.transact_commission = round((transact_price*transact_amount) * (self.commission_percent), self._base_precision)
        trade.transact_commission_percent=self.commission_percent
        trade.transact_total = round((transact_price * transact_amount) * (1 + self.commission_percent), self._base_precision)

        if self._is_valid_trade(trade):
            trade.valid = True

            if not trade.is_hold:
                trade.executed = True

            self._update_account(trade)
        else:
            trade.valid = False
            trade.executed = False
        return trade

    def reset(self):
        super().reset()

        self._current_step = 0
        self._balance = self.initial_balance
        self._portfolio = {self.base_instrument: self.balance}
        self._trades = pd.DataFrame([], columns=['step', 'symbol', 'type', 'amount', 'price'])
        self._performance = pd.DataFrame([], columns=['balance', 'net_worth', 'action', 'step'])
