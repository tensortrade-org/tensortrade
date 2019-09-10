# Copyright 2017 reinforce.io. All Rights Reserved.
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
# ==============================================================================

"""
OpenAI Gym Integration: https://gym.openai.com/.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import numpy as np
from tensorforce import TensorforceError
from tensorforce.environments import Environment


class OpenAIGym(Environment):
    """
    Bindings for OpenAIGym environment https://github.com/openai/gym
    To use install with "pip install gym". 
    """

    def __init__(self, gym_id, monitor=None, monitor_safe=False, monitor_video=0, visualize=False):
        """
        Initialize OpenAI Gym.
        Args:
            gym_id: OpenAI Gym environment ID. See https://gym.openai.com/envs
            monitor: Output directory. Setting this to None disables monitoring.
            monitor_safe: Setting this to True prevents existing log files to be overwritten. Default False.
            monitor_video: Save a video every monitor_video steps. Setting this to 0 disables recording of videos.
            visualize: If set True, the program will visualize the trainings of gym's environment. Note that such
                visualization is probabily going to slow down the training.
        """

        self.gym_id = gym_id
        self.gym = gym.make(gym_id)  # Might raise gym.error.UnregisteredEnv or gym.error.DeprecatedEnv
        self.visualize = visualize

        if monitor:
            if monitor_video == 0:
                video_callable = False
            else:
                video_callable = (lambda x: x % monitor_video == 0)
            self.gym = gym.wrappers.Monitor(self.gym, monitor, force=not monitor_safe, video_callable=video_callable)

        self._states = OpenAIGym.state_from_space(space=self.gym.observation_space)
        self._actions = OpenAIGym.action_from_space(space=self.gym.action_space)

    def __str__(self):
        return 'OpenAIGym({})'.format(self.gym_id)

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions

    def close(self):
        self.gym.close()
        self.gym = None

    def reset(self):
        if isinstance(self.gym, gym.wrappers.Monitor):
            self.gym.stats_recorder.done = True
        state = self.gym.reset()
        return OpenAIGym.flatten_state(state=state)

    def execute(self, action):
        if self.visualize:
            self.gym.render()
        action = OpenAIGym.unflatten_action(action=action)
        state, reward, terminal, _ = self.gym.step(action)
        return OpenAIGym.flatten_state(state=state), terminal, reward

    @staticmethod
    def state_from_space(space):
        if isinstance(space, gym.spaces.Discrete):
            return dict(shape=(), type='int')
        elif isinstance(space, gym.spaces.MultiBinary):
            return dict(shape=space.n, type='int')
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return dict(shape=space.num_discrete_space, type='int')
        elif isinstance(space, gym.spaces.Box):
            return dict(shape=tuple(space.shape), type='float')
        elif isinstance(space, gym.spaces.Tuple):
            states = dict()
            n = 0
            for n, space in enumerate(space.spaces):
                state = OpenAIGym.state_from_space(space=space)
                if 'type' in state:
                    states['gymtpl{}'.format(n)] = state
                else:
                    for name, state in state.items():
                        states['gymtpl{}-{}'.format(n, name)] = state
            return states
        elif isinstance(space, gym.spaces.Dict):
            states = dict()
            for space_name, space in space.spaces.items():
                state = OpenAIGym.state_from_space(space=space)
                if 'type' in state:
                    states[space_name] = state
                else:
                    for name, state in state.items():
                        states['{}-{}'.format(space_name, name)] = state
            return states
        else:
            raise TensorForceError('Unknown Gym space.')

    @staticmethod
    def flatten_state(state):
        if isinstance(state, tuple):
            states = dict()
            for n, state in enumerate(state):
                state = OpenAIGym.flatten_state(state=state)
                if isinstance(state, dict):
                    for name, state in state.items():
                        states['gymtpl{}-{}'.format(n, name)] = state
                else:
                    states['gymtpl{}'.format(n)] = state
            return states
        elif isinstance(state, dict):
            states = dict()
            for state_name, state in state.items():
                state = OpenAIGym.flatten_state(state=state)
                if isinstance(state, dict):
                    for name, state in state.items():
                        states['{}-{}'.format(state_name, name)] = state
                else:
                    states['{}'.format(state_name)] = state
            return states
        else:
            return state

    @staticmethod
    def action_from_space(space):
        if isinstance(space, gym.spaces.Discrete):
            return dict(type='int', num_actions=space.n)
        elif isinstance(space, gym.spaces.MultiBinary):
            return dict(type='bool', shape=space.n)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            num_discrete_space = len(space.nvec)
            if (space.nvec == space.nvec[0]).all():
                return dict(type='int', num_actions=space.nvec[0], shape=num_discrete_space)
            else:
                actions = dict()
                for n in range(num_discrete_space):
                    actions['gymmdc{}'.format(n)] = dict(type='int', num_actions=space.nvec[n])
                return actions
        elif isinstance(space, gym.spaces.Box):
            if (space.low == space.low[0]).all() and (space.high == space.high[0]).all():
                return dict(type='float', shape=space.low.shape,
                            min_value=np.float32(space.low[0]),
                            max_value=np.float32(space.high[0]))
            else:
                actions = dict()
                low = space.low.flatten()
                high = space.high.flatten()
                for n in range(low.shape[0]):
                    actions['gymbox{}'.format(n)] = dict(type='float', min_value=low[n], max_value=high[n])
                return actions
        elif isinstance(space, gym.spaces.Tuple):
            actions = dict()
            n = 0
            for n, space in enumerate(space.spaces):
                action = OpenAIGym.action_from_space(space=space)
                if 'type' in action:
                    actions['gymtpl{}'.format(n)] = action
                else:
                    for name, action in action.items():
                        actions['gymtpl{}-{}'.format(n, name)] = action
            return actions
        elif isinstance(space, gym.spaces.Dict):
            actions = dict()
            for space_name, space in space.spaces.items():
                action = OpenAIGym.action_from_space(space=space)
                if 'type' in action:
                    actions[space_name] = action
                else:
                    for name, action in action.items():
                        actions['{}-{}'.format(space_name, name)] = action
            return actions

        else:
            raise TensorForceError('Unknown Gym space.')

    @staticmethod
    def unflatten_action(action):
        if not isinstance(action, dict):
            return action
        elif all(name.startswith('gymmdc') for name in action) or \
                all(name.startswith('gymbox') for name in action) or \
                all(name.startswith('gymtpl') for name in action):
            space_type = next(iter(action))[:6]
            actions = list()
            n = 0
            while True:
                if any(name.startswith(space_type + str(n) + '-') for name in action):
                    inner_action = {
                        name[name.index('-') + 1:] for name, inner_action in action.items()
                        if name.startswith(space_type + str(n))
                    }
                    actions.append(OpenAIGym.unflatten_action(action=inner_action))
                elif any(name == space_type + str(n) for name in action):
                    actions.append(action[space_type + str(n)])
                else:
                    break
                n += 1
            return tuple(actions)
        else:
            actions = dict()
            for name, action in action.items():
                if '-' in name:
                    name, inner_name = name.split('-', 1)
                    if name not in actions:
                        actions[name] = dict()
                    actions[name][inner_name] = action
                else:
                    actions[name] = action
            for name, action in actions.items():
                if isinstance(action, dict):
                    actions[name] = OpenAIGym.unflatten_action(action=action)
            return actions