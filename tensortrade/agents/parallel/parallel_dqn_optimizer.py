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


import tensorflow as tf

from multiprocessing import Process, Queue

from tensortrade.agents import ReplayMemory, DQNTransition


class ParallelDQNOptimizer(Process):
    def __init__(self,
                 model: 'ParallelDQNModel',
                 n_envs: int,
                 memory_queue: Queue,
                 model_update_queue: Queue,
                 done_queue: Queue,
                 discount_factor: float = 0.9999,
                 batch_size: int = 128,
                 learning_rate: float = 0.0001,
                 memory_capacity: int = 10000):
        super().__init__()

        self.model = model
        self.n_envs = n_envs
        self.memory_queue = memory_queue
        self.model_update_queue = model_update_queue
        self.done_queue = done_queue
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.memory_capacity = memory_capacity

    def run(self):
        memory = ReplayMemory(self.memory_capacity, transition_type=DQNTransition)

        optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        loss_fn = tf.keras.losses.Huber()

        while self.done_queue.qsize() < self.n_envs:
            while self.memory_queue.qsize() > 0:
                sample = self.memory_queue.get()
                memory.push(*sample)

            if len(memory) < self.batch_size:
                continue

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

            self.model_update_queue.put(self.model)
