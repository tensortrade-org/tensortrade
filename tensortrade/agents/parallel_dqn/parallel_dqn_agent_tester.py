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
# limitations under the License.

"""
References:
    - https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998
    - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm
"""

import numpy as np
from collections import namedtuple

from tensortrade.agents.parallel_dqn.parallel_dqn_model import ParallelDQNModel

DQNTransition = namedtuple('DQNTransition', ['state', 'action', 'reward', 'next_state', 'done'])

class ParallelDQNAgentTester:

    def __init__(self,
                 env: 'TradingEnvironment',
                 model: ParallelDQNModel):

        self.env = env
        self.n_actions = env.action_space.n
        self.observation_shape = env.observation_space.shape
        self.model = model

    def test(self,
              n_steps: int = None,
              **kwargs) -> float:
        eps_start: float = kwargs.get('eps_start', 0.9)
        eps_end: float = kwargs.get('eps_end', 0.05)
        eps_decay_steps: int = kwargs.get('eps_decay_steps', 200)

        render_interval = 50
        episode = 0
        total_reward = 0

        self.env.max_steps = n_steps

        state = self.env.reset()
        done = False
        steps_done = 0

        while not done:
            threshold = eps_end + (eps_start - eps_end) * np.exp(-steps_done / eps_decay_steps)
            action = self.model.get_action(state, threshold=threshold)
            next_state, reward, done, _ = self.env.step(action)

            state = next_state
            total_reward += reward
            steps_done += 1

            if n_steps and steps_done >= n_steps:
                done = True

            if render_interval is not None and steps_done % render_interval == 0:
                self.env.render(episode)

        if not render_interval or steps_done < n_steps:
            self.env.render(episode)  # render final state at episode end if not rendered earlier

        mean_reward = total_reward / steps_done

        return mean_reward
