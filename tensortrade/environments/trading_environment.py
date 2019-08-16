import gym
import logging
import pandas as pd
import numpy as np

from gym import spaces
from typing import Union, Tuple

from tensortrade.environments.actions import ActionStrategy, TradeType
from tensortrade.environments.rewards import RewardStrategy
from tensortrade.exchanges import AssetExchange


class TradingEnvironment(gym.Env):
    """A trading environment made for use with Gym-compatible reinforcement learning algorithms."""

    def __init__(self,
                 action_strategy: ActionStrategy,
                 reward_strategy: RewardStrategy,
                 exchange: AssetExchange,
                 **kwargs):
        """
        Args:
            action_strategy:  The strategy for transforming an action into a `TradeDefinition` at each timestep.
            reward_strategy: The strategy for determining the reward at each timestep.
            exchange: The `AssetExchange` that will be used to feed data from and execute trades within.
            kwargs (optional): Additional arguments for tuning the environment, logging, etc.
        """

        super().__init__()

        self.action_strategy = action_strategy
        self.reward_strategy = reward_strategy
        self.exchange = exchange

        self.action_space_dtype: type = kwargs.get('action_space_dtype', np.float16)
        self.observation_space_dtype: type = kwargs.get('observation_space_dtype', np.float16)

        self.action_strategy.set_dtype(self.action_space_dtype)
        self.exchange.set_dtype(self.observation_space_dtype)

        self.action_space = self.action_strategy.action_space()
        self.observation_space = self.exchange.observation_space()

        self.logger_name: int = kwargs.get('logger_name', __name__)
        self.log_level: int = kwargs.get('log_level', logging.DEBUG)
        self.disable_tensorflow_logs: bool = kwargs.get('disable_tensorflow_logger', True)

        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(self.log_level)

        logging.getLogger('tensorflow').disabled = self.disable_tensorflow_logs

    def _take_action(self, action: Union[int, float]):
        """Determines a specific trade to be taken and executes it within the exchange.

        Args:
            action: The trade action provided by the agent for this timestep.
        """
        symbol, trade_type, amount, price = self.action_strategy.get_trade(
            action=action, exchange=self.exchange)

        self.exchange.execute_trade(symbol=symbol, trade_type=trade_type,
                                    amount=amount, price=price)

    def _next_observation(self) -> pd.DataFrame:
        """Returns the next observation from the exchange.

        Returns:
            observation: Provided by the environment's exchange, often OHLCV or tick trade history data points.
        """
        self.current_step += 1

        observation = self.exchange.next_observation()
        observation = observation.fillna(0, axis=1)

        return observation

    def _get_reward(self) -> float:
        """Returns the reward for the current timestep.

        Returns:
            done: If `True`, the environment is complete and should be restarted.
        """
        reward: float = self.reward_strategy.get_reward(
            current_step=self.current_step, exchange=self.exchange)

        return reward if np.isfinite(reward) else 0

    def _done(self) -> bool:
        """Returns whether or not the environment is done this timestep.

        Returns:
            done: If `True`, the environment is complete and should be restarted.
        """
        lost_90_percent_net_worth = self.exchange.profit_loss_percent() < 0.1
        has_next_obs: bool = self.exchange.has_next_observation()

        return lost_90_percent_net_worth or not has_next_obs

    def _info(self) -> dict:
        """Returns any auxiliary, diagnostic, or debugging information for the current timestep.

        Returns:
            info: A dictionary containing the exchange used and the current timestep.
        """
        return {'current_step': self.current_step, 'exchange': self.exchange}

    def step(self, action) -> Tuple[pd.DataFrame, float, bool, dict]:
        """Run one timestep within the environment based on the specified action.

        Args:
            action: The trade action provided by the agent for this timestep.

        Returns:
            observation (pandas.DataFrame): Provided by the environment's exchange, often OHLCV or tick trade history data points.
            reward (float): An amount corresponding to the advantage gained by the agent in this timestep.
            done (bool): If `True`, the environment is complete and should be restarted.
            info (dict): Any auxiliary, diagnostic, or debugging information to output.
        """

        self._take_action(action)

        observation = self._next_observation()
        reward = self._get_reward()
        done = self._done()
        info = self._info()

        return observation, reward, done, info

    def reset(self) -> pd.DataFrame:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            observation: the initial observation.
        """

        self.action_strategy.reset()
        self.reward_strategy.reset()
        self.exchange.reset()

        self.current_step = 0

        return self._next_observation()

    def render(self, mode='none'):
        """Renders the environment."""
        pass
