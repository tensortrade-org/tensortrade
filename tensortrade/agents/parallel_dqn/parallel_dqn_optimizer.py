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
import time
from multiprocessing import Queue

import numpy as np
import tensorflow as tf

from tensortrade.agents import ReplayMemory, DQNTransition


class ParallelDQNOptimizer:
    def __init__(self,
                 create_env_func,
                 model: 'ParallelDQNModel',
                 n_envs: int,
                 n_steps: int,
                 n_episodes: int,
                 memory_queue: Queue,
                 model_update_queue: Queue,
                 sync_queue: Queue,
                 done_queue: Queue,
                 discount_factor: float = 0.9999,
                 batch_size: int = 128,
                 learning_rate: float = 0.0001,
                 memory_capacity: int = 10000,
                 eps_end: int = 0.05,
                 eps_start: int = 0.99,
                 eps_decay_steps: int = 2000):

        self.create_env_func = create_env_func
        self.model = model
        self.n_envs = n_envs
        self.n_episodes = n_episodes
        self.n_steps = n_steps or np.iinfo(np.int32).max
        self.memory_queue = memory_queue
        self.model_update_queue = model_update_queue
        self.sync_queue = sync_queue
        self.done_queue = done_queue
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.memory_capacity = memory_capacity
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.eps_decay_steps = eps_decay_steps

    def empty_queue(self, queue: Queue):
        while queue.qsize() > 0:
            queue.get()

    def run(self):
        episode = 0

        for i in range(self.n_envs):
            self.model_update_queue.put(self.model.policy_network.to_json())

        memory = ReplayMemory(self.memory_capacity, transition_type=DQNTransition)

        optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        loss_fn = tf.keras.losses.Huber()

        while self.done_queue.qsize() < self.n_envs:
            while self.sync_queue.qsize() < self.n_envs:
                while self.memory_queue.qsize() > 0:
                    sample = self.memory_queue.get()
                    memory.push(*sample)

            self.empty_queue(self.sync_queue)

            transitions = memory.sample(self.batch_size)
            batch = DQNTransition(*zip(*transitions))

            state_batch = tf.convert_to_tensor(batch.state)
            action_batch = tf.convert_to_tensor(batch.action)
            reward_batch = tf.convert_to_tensor(batch.reward, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(batch.next_state)
            done_batch = tf.convert_to_tensor(batch.done)

            with tf.GradientTape() as tape:
                state_action_values = tf.math.reduce_sum(
                    self.model.policy_network(state_batch) *
                    tf.one_hot(action_batch, self.model.n_actions),
                    axis=1
                )

                next_state_values = tf.where(
                    done_batch,
                    tf.zeros(self.batch_size),
                    tf.math.reduce_max(self.model.target_network(next_state_batch), axis=1)
                )

                expected_state_action_values = reward_batch + \
                                               (self.discount_factor * next_state_values)
                loss_value = loss_fn(expected_state_action_values, state_action_values)

            variables = self.model.policy_network.trainable_variables
            gradients = tape.gradient(loss_value, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            self.model.update_target_network()

            json_model = self.model.policy_network.to_json()
            for n in range(self.n_envs):
                self.model_update_queue.put(json_model)

            for n in range(self.n_envs*2):
                self.sync_queue.put(1)

            episode += 1
            if episode == self.n_episodes:
                break

            while self.model_update_queue.qsize() > 0:
                time.sleep(.1)

            print("")
            self.empty_queue(self.sync_queue)

        self.empty_queue(self.model_update_queue)

        time.sleep(2)

        return self.model
