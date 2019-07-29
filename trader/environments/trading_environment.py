import gym
import logging
import pandas as pd
import numpy as np

from gym import spaces

from trader.actions import ActionStrategy
from trader.rewards import RewardStrategy
from trader.exchanges import AssetExchange


class TradeType(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2


class TradingEnvironment(gym.Env):
    '''A trading environment made for use with Gym-compatible reinforcement learning algorithms'''

    def __init__(self,
                 action_strategy: ActionStrategy,
                 reward_strategy: RewardStrategy,
                 exchange: AssetExchange,
                 **kwargs):
        super(TradingEnvironment, self).__init__()

        self.action_strategy = action_strategy
        self.reward_strategy = reward_strategy
        self.exchange = exchange

        self.space_dtype: type = kwargs.get('space_dtype', np.float16)
        self.initial_balance: float = kwargs.get('initial_balance', 10000.0)
        self.max_allowed_slippage_percent: float = kwargs.get('max_allowed_slippage_percent', 3.0)

        self.action_strategy.set_dtype(self.space_dtype)
        self.reward_strategy.set_dtype(self.space_dtype)
        self.exchange.set_dtype(self.space_dtype)
        self.exchange.set_max_allowed_slippage_percent(self.max_allowed_slippage_percent)

        self.action_space = self.action_strategy.action_space()
        self.observation_space = self.exchange.observation_space()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(kwargs.get('log_level', logging.DEBUG))

        logging.getLogger('tensorflow').disabled = kwargs.get('disable_tensorflow_logger', True)

    def _sum_net_worth(self):
        net_worth = self.balance

        for symbol, amount in vars(net_worth).items():
            current_price = self.exchange.current_price(symbol=symbol)
            net_worth += current_price * amount

        return net_worth

    def _update_account(self, symbol: str, trade_type: TradeType, fill_amount: float, fill_price: float):
        self.trades.append({
            'step': self.current_step,
            'symbol': symbol,
            'type': trade_type,
            'amount': fill_amount,
            'price': fill_price
        })

        if trade_type is TradeType.BUY:
            self.balance -= fill_amount * fill_price
            self.assets_held[symbol] += fill_amount
        elif trade_type is TradeType.SELL:
            self.balance += fill_amount * fill_price
            self.assets_held[symbol] -= fill_amount

        self.net_worth = self._sum_net_worth()

    def _take_action(self, action: int | float):
        symbol, trade_type, amount, price = self.action_strategy.suggest_trade(action=action,
                                                                               balance=self.balance,
                                                                               assets_held=self.assets_held,
                                                                               exchange=self.exchange)

        if trade_type is TradeType.HOLD or amount is 0:
            return

        fill_amount, fill_price = self.exchange.execute_trade(symbol=symbol,
                                                              trade_type=trade_type,
                                                              amount=amount,
                                                              price=price)

        if fill_amount > 0:
            self._update_account(symbol=symbol,
                                 trade_type=trade_type,
                                 fill_amount=fill_amount,
                                 fill_price=fill_price)

    def _next_observation(self):
        self.current_step += 1

        obs = self.exchange.next_observation()
        obs[np.bitwise_not(np.isfinite(obs))] = 0

        return obs

    def _reward(self):
        reward: float = self.reward_strategy.reward(current_step=self.current_step,
                                                    balance=self.balance,
                                                    net_worth=self.net_worth,
                                                    assets_held=self.assets_held,
                                                    trades=self.trades,
                                                    performance=self.performance)

        return reward if np.isfinite(reward) else 0

    def _done(self):
        lost_90_percent_net_worth = self.net_worth < self.initial_balance / 10
        has_next_obs: bool = self.exchange.has_next_observation()

        return lost_90_percent_net_worth or not has_next_obs

    def _info(self):
        return {'assets_held': assets_held, 'trades': self.trades, 'performance': self.performance}

    def step(self, action):
        self._take_action(action)

        obs = self._next_observation()
        reward = self._reward()
        done = self._done()
        info = self._info()

        self.performance.append({
            'step': self.current_step,
            'reward': reward,
            'balance': self.balance,
            'net_worth': self.net_worth
        })

        return obs, reward, done, info

    def _reset_account(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance

        self.assets_held = {}

        self.trades = pd.DataFrame([], columns=['step', 'symbol', 'type', 'amount', 'price'])

        self.performance = pd.DataFrame([{
            'step': 0,
            'reward': 0,
            'balance': self.balance,
            'net_worth': self.net_worth,
        }])

    def reset(self):
        self.action_strategy.reset()
        self.reward_strategy.reset()
        self.exchange.reset()

        self._reset_account()

        return self._next_observation()

    def render(self, mode='none'):
        pass
