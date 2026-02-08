# Copyright 2020 The TensorTrade Authors.
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

import logging
import uuid
from random import randint
from typing import Any

import gymnasium as gym
import numpy as np

from tensortrade.core import Clock, Component, TimeIndexed
from tensortrade.env.generic import (
    ActionScheme,
    Informer,
    Observer,
    Renderer,
    RewardScheme,
    Stopper,
)


class TradingEnv(gym.Env, TimeIndexed):
    """A trading environment made for use with Gym-compatible reinforcement learning algorithms.

    Parameters
    ----------
    action_scheme : `ActionScheme`
        A component for generating an action to perform at each step of the
        environment.
    reward_scheme : `RewardScheme`
        A component for computing reward after each step of the environment.
    observer : `Observer`
        A component for generating observations after each step of the
        environment.
    informer : `Informer`
        A component for providing information after each step of the
        environment.
    renderer : `Renderer`
        A component for rendering the environment.
    kwargs : keyword arguments
        Additional keyword arguments needed to create the environment.
    """

    agent_id: str | None = None
    episode_id: str | None = None

    def __init__(
        self,
        action_scheme: ActionScheme,
        reward_scheme: RewardScheme,
        observer: Observer,
        stopper: Stopper,
        informer: Informer,
        renderer: Renderer,
        min_periods: int | None = None,
        max_episode_steps: int | None = None,
        random_start_pct: float = 0.00,
        device: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.clock = Clock()

        self.action_scheme = action_scheme
        self.reward_scheme = reward_scheme
        self.observer = observer
        self.stopper = stopper
        self.informer = informer
        self.renderer = renderer
        self.min_periods = min_periods
        self.max_episode_steps = max_episode_steps
        self.random_start_pct = random_start_pct
        self.device = device

        # Register the environment in Gym and fetch spec
        gym.register(
            id="TensorTrade-v0",
            entry_point=lambda: self,
            max_episode_steps=max_episode_steps,
        )
        self.spec = gym.spec(env_id="TensorTrade-v0")

        for c in self.components.values():
            c.clock = self.clock

        self.action_space = action_scheme.action_space
        self.observation_space = observer.observation_space

        self._enable_logger = kwargs.get("enable_logger", False)
        if self._enable_logger:
            self.logger = logging.getLogger(kwargs.get("logger_name", __name__))
            self.logger.setLevel(kwargs.get("log_level", logging.DEBUG))

    def _ensure_numpy(self, obs: Any) -> np.ndarray:
        """Ensure observation is returned as numpy array for GPU compatibility.

        Parameters
        ----------
        obs : Any
            The observation to convert

        Returns
        -------
        np.ndarray
            The observation as a numpy array
        """
        if hasattr(obs, "cpu"):  # PyTorch tensor
            return obs.cpu().numpy()
        elif hasattr(obs, "numpy"):  # TensorFlow tensor
            return obs.numpy()
        elif isinstance(obs, np.ndarray):
            return obs
        else:
            return np.array(obs)

    @property
    def components(self) -> "dict[str, Component]":
        """Get the components of the environment (`Dict[str,Component]`, read-only)."""
        return {
            "action_scheme": self.action_scheme,
            "reward_scheme": self.reward_scheme,
            "observer": self.observer,
            "stopper": self.stopper,
            "informer": self.informer,
            "renderer": self.renderer,
        }

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Make one step through the environment.

        Parameters
        ----------
        action : Any
            An action to perform on the environment.

        Returns
        -------
        `np.array`
            The observation of the environment after the action being
            performed.
        float
            The computed reward for performing the action.
        bool
            Whether or not the episode is terminated (goal reached or failed).
        bool
            Whether or not the episode is truncated (time limit or other).
        dict
            The information gathered after completing the step.
        """
        self.action_scheme.perform(self, action)

        obs = self.observer.observe(self)
        # Ensure observation is numpy array for GPU compatibility
        obs = self._ensure_numpy(obs)
        reward = self.reward_scheme.reward(self)
        terminated = self.stopper.stop(self)
        truncated = False  # Truncation is handled by gymnasium's TimeLimit wrapper
        info = self.informer.info(self)

        self.clock.increment()

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        Parameters
        ----------
        seed : int, optional
            The seed for the random number generator.
        options : dict, optional
            Additional options for resetting the environment.

        Returns
        -------
        obs : `np.array`
            The first observation of the environment.
        info : dict
            Additional information from the reset.
        """
        super().reset(seed=seed)

        if self.random_start_pct > 0.00:
            size = len(self.observer.feed.process[-1].inputs[0].iterable)
            random_start = randint(0, int(size * self.random_start_pct))
        else:
            random_start = 0

        self.episode_id = str(uuid.uuid4())
        self.clock.reset()

        for c in self.components.values():
            if hasattr(c, "reset"):
                if isinstance(c, Observer):
                    c.reset(random_start=random_start)
                else:
                    c.reset()

        obs = self.observer.observe(self)
        # Ensure observation is numpy array for GPU compatibility
        obs = self._ensure_numpy(obs)
        info = self.informer.info(self)

        self.clock.increment()

        return obs, info

    def render(self, **kwargs) -> None:
        """Render the environment."""
        self.renderer.render(self, **kwargs)

    def save(self) -> None:
        """Save the rendered view of the environment."""
        self.renderer.save()

    def close(self) -> None:
        """Close the environment."""
        self.renderer.close()
