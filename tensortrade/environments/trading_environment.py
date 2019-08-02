import gym
import logging
import pandas as pd
import numpy as np

from gym import spaces
from typing import Union

from tensortrade.actions import ActionStrategy, TradeType
from tensortrade.rewards import RewardStrategy
from tensortrade.exchanges import AssetExchange


class TradingEnvironment(gym.Env):
    '''A trading environment made for use with Gym-compatible reinforcement learning algorithms

        The methods that are accessed publicly are "step", "reset" and "render"
    '''

    def __init__(self,
                 action_strategy: ActionStrategy,
                 reward_strategy: RewardStrategy,
                 exchange: AssetExchange,
                 **kwargs):
        '''Create the trading environment

            # Arguments
                action_strategy:  the action strategy
                reward_strategy: the reward strategy
                exchange: the AssetExchange object, that should be used
                kwargs: additional arguments, to define log levels, floating point precision,

        '''

        super().__init__()

        self.action_strategy = action_strategy
        self.reward_strategy = reward_strategy
        self.exchange = exchange

        self.space_dtype: type = kwargs.get('space_dtype', np.float16)
        self.logger_name: int = kwargs.get('logger_name', __name__)
        self.log_level: int = kwargs.get('log_level', logging.DEBUG)

        self.action_strategy.set_dtype(self.space_dtype)
        self.reward_strategy.set_dtype(self.space_dtype)
        self.exchange.set_dtype(self.space_dtype)

        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(self.log_level)

        self.action_space = self.action_strategy.action_space()
        self.observation_space = self.exchange.observation_space()

        logging.getLogger('tensorflow').disabled = kwargs.get('disable_tensorflow_logger', True)

    def _take_action(self, action: Union[int, float]):
        symbol, trade_type, amount, price = self.action_strategy.suggest_trade(action=action, exchange=self.exchange)

        self.exchange.execute_trade(symbol=symbol, trade_type=trade_type, amount=amount, price=price)

    def _next_observation(self):
        self.current_step += 1

        obs = self.exchange.next_observation()
        obs[np.bitwise_not(np.isfinite(obs))] = 0

        return obs

    def _reward(self):
        reward: float = self.reward_strategy.reward(current_step=self.current_step, exchange=self.exchange)

        return reward if np.isfinite(reward) else 0

    def _done(self):
        lost_90_percent_net_worth = self.exchange.profit_loss_percent() < 0.1
        has_next_obs: bool = self.exchange.has_next_observation()

        return lost_90_percent_net_worth or not has_next_obs

    def _info(self):
        return {'current_step': self.current_step, 'exchange': self.exchange}

    def step(self, action):
        '''Run one timestep of the environment's dynamics. When end of
            episode is reached, you are responsible for calling `reset()`
            to reset this environment's state.

            Accepts an action and returns a tuple (observation, reward, done, info).

            # Arguments:
                action (object): an action provided by the agent

            # Returns:
                observation (Dataframe): provided by the exchange, usually normalized ohlc and volume information
                reward (float) : amount of reward returned after previous action
                done (bool): whether the episode has ended, in which case further step() calls will return undefined results
                info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        '''

        self._take_action(action)

        obs = self._next_observation()
        reward = self._reward()
        done = self._done()
        info = self._info()

        return obs, reward, done, info

    def reset(self):
        '''Resets the state of the environment and returns an initial observation.

            # Returns:
                observation: the initial observation.
        '''

        self.action_strategy.reset()
        self.reward_strategy.reset()
        self.exchange.reset()

        self.current_step = 0

        return self._next_observation()

    def render(self, mode='none'):
        '''Renders the environment.'''

        pass
