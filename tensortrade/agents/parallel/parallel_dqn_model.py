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


import random
import numpy as np
import tensorflow as tf

from typing import Callable


class ParallelDQNModel:

    def __init__(self,
                 create_env: Callable[[], 'TradingEnvironment'],
                 policy_network: tf.keras.Model = None):
        temp_env = create_env()

        self.n_actions = temp_env.action_space.n
        self.observation_shape = temp_env.observation_space.shape

        self.policy_network = policy_network or self._build_policy_network()

        self.target_network = tf.keras.models.clone_model(self.policy_network)
        self.target_network.trainable = False

    def _build_policy_network(self):
        network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.observation_shape),
            tf.keras.layers.Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.n_actions, activation="sigmoid"),
            tf.keras.layers.Dense(self.n_actions, activation="softmax")
        ])

        return network

    def restore(self, path: str, **kwargs):
        self.policy_network = tf.keras.models.load_model(path)
        self.target_network = tf.keras.models.clone_model(self.policy_network)
        self.target_network.trainable = False

    def save(self, path: str, **kwargs):
        agent_id: int = kwargs.get('agent_id', 'No_ID')
        episode: int = kwargs.get('episode', None)

        if episode:
            filename = "policy_network__" + agent_id + "__" + str(episode).zfill(3) + ".hdf5"
        else:
            filename = "policy_network__" + agent_id + ".hdf5"

        self.policy_network.save(path + filename)

    def get_action(self, state: np.ndarray, **kwargs) -> int:
        threshold: float = kwargs.get('threshold', 0)

        rand = random.random()

        if rand < threshold:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.policy_network(np.expand_dims(state, 0)))

    def update_networks(self, model: 'ParallelDQNModel'):
        self.policy_network.set_weights(model.policy_network.get_weights())
        self.target_network.set_weights(model.target_network.get_weights())

    def update_target_network(self):
        self.target_network.set_weights(self.policy_network.get_weights())
