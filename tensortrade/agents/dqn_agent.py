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
# limitations under the License.

from deprecated import deprecated
import random
import numpy as np
import tensorflow as tf
from collections import namedtuple

from tensortrade.agents import Agent, ReplayMemory
from datetime import datetime


DQNTransition = namedtuple('DQNTransition', ['state', 'action', 'reward', 'next_state', 'done'])


@deprecated(version='1.0.4', reason="Builtin agents are being deprecated in favor of external implementations (ie: Ray)")
class DQNAgent(Agent):
    """

    References:
    ===========
        - https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998
        - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm
        - https://arxiv.org/abs/1802.00308
    """

    def __init__(self,
                 env: 'TradingEnv',
                 policy_network: tf.keras.Model = None):
        self.env = env
        self.n_actions = env.action_space.n
        self.observation_shape = env.observation_space.shape

        self.policy_network = policy_network or self._build_policy_network()

        self.target_network = tf.keras.models.clone_model(self.policy_network)
        self.target_network.trainable = False

        self.env.agent_id = self.id

    def _build_policy_network(self):
        '''
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
        '''

        dropout = 0.9
        stride = 1

        inputs = tf.keras.layers.Input(shape=self.observation_shape, 
                                       dtype='float32')

        pad_1 = tf.keras.layers.ZeroPadding1D(padding=stride)(inputs)

        conv_1 = tf.keras.layers.Conv1D(filters=16, 
                                        kernel_size=4, 
                                        strides=2, 
                                        padding='causal', 
                                        activation=tf.keras.layers.PReLU(), 
                                        kernel_initializer='he_uniform', 
                                        use_bias=True)(pad_1)

        norm_1 = tf.keras.layers.BatchNormalization()(conv_1)

        conv_2 = tf.keras.layers.Conv1D(filters=32, 
                                        kernel_size=4, 
                                        strides=2, 
                                        padding='causal', 
                                        activation=tf.keras.layers.PReLU(), 
                                        kernel_initializer='he_uniform', 
                                        use_bias=True)(pad_1)

        norm_2 = tf.keras.layers.BatchNormalization()(conv_2)

        conv_3 = tf.keras.layers.Conv1D(filters=64, 
                                        kernel_size=4, 
                                        strides=2, 
                                        padding='causal', 
                                        activation=tf.keras.layers.PReLU(), 
                                        kernel_initializer='he_uniform', 
                                        use_bias=True)(pad_1)

        norm_3 = tf.keras.layers.BatchNormalization()(conv_3)

        concat_conv_1 = tf.keras.layers.Concatenate()([norm_1, norm_2, norm_3])

        dropout_1 = tf.keras.layers.Dropout(rate=dropout)(concat_conv_1)

        conv_4 = tf.keras.layers.Conv1D(filters=16, 
                                        kernel_size=4, 
                                        strides=2, 
                                        padding='causal', 
                                        activation=tf.keras.layers.PReLU(), 
                                        kernel_initializer='he_uniform', 
                                        use_bias=True)(dropout_1)

        norm_4 = tf.keras.layers.BatchNormalization()(conv_4)

        conv_5 = tf.keras.layers.Conv1D(filters=32, 
                                        kernel_size=4, 
                                        strides=2, 
                                        padding='causal', 
                                        activation=tf.keras.layers.PReLU(), 
                                        kernel_initializer='he_uniform', 
                                        use_bias=True)(dropout_1)

        norm_5 = tf.keras.layers.BatchNormalization()(conv_5)

        conv_6 = tf.keras.layers.Conv1D(filters=64, 
                                        kernel_size=4, 
                                        strides=2, 
                                        padding='causal', 
                                        activation=tf.keras.layers.PReLU(), 
                                        kernel_initializer='he_uniform', 
                                        use_bias=True)(dropout_1)

        norm_6 = tf.keras.layers.BatchNormalization()(conv_6)

        concat_conv_2 = tf.keras.layers.Concatenate()([norm_4, norm_5, norm_6])

        dropout_2 = tf.keras.layers.Dropout(rate=dropout)(concat_conv_2)

        conv_7 = tf.keras.layers.Conv1D(filters=16, 
                                        kernel_size=4, 
                                        strides=2, 
                                        padding='causal', 
                                        activation=tf.keras.layers.PReLU(), 
                                        kernel_initializer='he_uniform', 
                                        use_bias=True)(dropout_2)

        norm_7 = tf.keras.layers.BatchNormalization()(conv_7)

        conv_8 = tf.keras.layers.Conv1D(filters=32, 
                                         kernel_size=4, 
                                         strides=2, 
                                         padding='causal', 
                                         activation=tf.keras.layers.PReLU(), 
                                         kernel_initializer='he_uniform', 
                                         use_bias=True)(dropout_2)

        norm_8 = tf.keras.layers.BatchNormalization()(conv_8)

        conv_9 = tf.keras.layers.Conv1D(filters=64, 
                                         kernel_size=4, 
                                         strides=2, 
                                         padding='causal', 
                                         activation=tf.keras.layers.PReLU(), 
                                         kernel_initializer='he_uniform', 
                                         use_bias=True)(dropout_2)

        norm_9 = tf.keras.layers.BatchNormalization()(conv_9)

        concat_conv_3 = tf.keras.layers.Concatenate()([norm_7, 
                                                       norm_8, 
                                                       norm_9])

        dropout_3 = tf.keras.layers.Dropout(rate=dropout)(concat_conv_3)

        pool_1 = tf.keras.layers.AveragePooling1D(pool_size=3, strides=2)(dropout_3)

        gru_1 = tf.keras.layers.GRU(units=64, 
                                    activation='tanh', 
                                    return_sequences=True)(pool_1)

        dropout_4 = tf.keras.layers.Dropout(rate=dropout)(gru_1)

        gru_2 = tf.keras.layers.GRU(units=64, 
                                    activation='tanh', 
                                    return_sequences=True)(dropout_4)

        dropout_5 = tf.keras.layers.Dropout(rate=dropout)(gru_2)

        concat_rnn_1 = tf.keras.layers.Concatenate()([dropout_4, dropout_5])

        gru_3 = tf.keras.layers.GRU(units=64, 
                                    activation='tanh', 
                                    return_sequences=True)(concat_rnn_1)

        dropout_6 = tf.keras.layers.Dropout(rate=dropout)(gru_3)

        concat_rnn_2 = tf.keras.layers.Concatenate()([dropout_4, dropout_5, dropout_6])

        gru_4 = tf.keras.layers.GRU(units=64, activation='tanh')(concat_rnn_2)

        flat_2 = tf.keras.layers.Flatten()(gru_4)

        dropout_7 = tf.keras.layers.Dropout(rate=dropout)(flat_2)

        dense_1 = tf.keras.layers.Dense(units=32, activation='softmax')(dropout_7)

        dense_2 = tf.keras.layers.Dense(units=16, activation='softmax')(dense_1)

        dense_3 = tf.keras.layers.Dense(units=16, 
                                        activation=tf.keras.layers.PReLU(), 
                                        kernel_initializer='he_uniform')(dense_2)

        dropout_8 = tf.keras.layers.Dropout(rate=dropout)(dense_3)

        #outputs = tf.keras.layers.Dense(units=self.n_actions, activation='linear')(dropout_8)
        pre_outputs = tf.keras.layers.Dense(units=self.n_actions, activation='sigmoid')(dropout_8)
        outputs = tf.keras.layers.Dense(units=self.n_actions, activation='softmax')(pre_outputs)

        network = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        return network

    def restore(self, path: str, **kwargs):
        self.policy_network = tf.keras.models.load_model(path)
        self.target_network = tf.keras.models.clone_model(self.policy_network)
        self.target_network.trainable = False

    def save(self, path: str, **kwargs):
        episode: int = kwargs.get('episode', None)

        if episode:
            filename = "policy_network__" + self.id[:7] + "__" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".hdf5"
        else:
            filename = "policy_network__" + self.id[:7] + "__" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".hdf5"

        self.policy_network.save(path + filename)

    def get_action(self, state: np.ndarray, **kwargs) -> int:
        threshold: float = kwargs.get('threshold', 0)

        rand = random.random()

        if rand < threshold:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.policy_network(np.expand_dims(state, 0)))

    def _apply_gradient_descent(self, memory: ReplayMemory, batch_size: int, learning_rate: float, discount_factor: float):

        # Optimization strategy.
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)

        loss = tf.keras.losses.Huber()

        transitions = memory.sample(batch_size)
        batch = DQNTransition(*zip(*transitions))

        state_batch = tf.convert_to_tensor(batch.state)
        action_batch = tf.convert_to_tensor(batch.action)
        reward_batch = tf.convert_to_tensor(batch.reward, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(batch.next_state)
        done_batch = tf.convert_to_tensor(batch.done)

        with tf.GradientTape() as tape:
            state_action_values = tf.math.reduce_sum(
                self.policy_network(state_batch) * tf.one_hot(action_batch, self.n_actions),
                axis=1
            )

            next_state_values = tf.where(
                done_batch,
                tf.zeros(batch_size),
                tf.math.reduce_max(self.target_network(next_state_batch), axis=1)
            )

            expected_state_action_values = reward_batch + (discount_factor * next_state_values)
            loss_value = loss(expected_state_action_values, state_action_values)

        variables = self.policy_network.trainable_variables
        gradients = tape.gradient(loss_value, variables)
        optimizer.apply_gradients(zip(gradients, variables))

    def train(self,
              n_steps: int = 1000,
              n_episodes: int = 10,
              save_every: int = None,
              save_path: str = 'agent/',
              callback: callable = None,
              **kwargs) -> float:
        batch_size: int = kwargs.get('batch_size', 256)
        memory_capacity: int = kwargs.get('memory_capacity', n_steps * 10)
        discount_factor: float = kwargs.get('discount_factor', 0.95)
        learning_rate: float = kwargs.get('learning_rate', 0.01)
        eps_start: float = kwargs.get('eps_start', 0.9)
        eps_end: float = kwargs.get('eps_end', 0.05)
        eps_decay_steps: int = kwargs.get('eps_decay_steps', n_steps)
        update_target_every: int = kwargs.get('update_target_every', 1000)
        render_interval: int = kwargs.get('render_interval', n_steps // 10)  # in steps, None for episode end renderers only

        memory = ReplayMemory(memory_capacity, transition_type=DQNTransition)
        episode = 0
        total_steps_done = 0
        total_reward = 0
        stop_training = False

        if n_steps and not n_episodes:
            n_episodes = np.iinfo(np.int32).max

        print('====      AGENT ID: {}      ===='.format(self.id))

        while episode < n_episodes and not stop_training:
            state = self.env.reset()
            done = False
            steps_done = 0

            while not done:
                threshold = eps_end + (eps_start - eps_end) * np.exp(-total_steps_done / eps_decay_steps)
                action = self.get_action(state, threshold=threshold)
                next_state, reward, done, _ = self.env.step(action)

                memory.push(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps_done += 1
                total_steps_done += 1

                if len(memory) < batch_size:
                    continue

                self._apply_gradient_descent(memory, batch_size, learning_rate, discount_factor)

                if n_steps and steps_done >= n_steps:
                    done = True

                if render_interval is not None and steps_done % render_interval == 0:
                    self.env.render(
                        episode=episode,
                        max_episodes=n_episodes,
                        max_steps=n_steps
                    )

                if steps_done % update_target_every == 0:
                    self.target_network = tf.keras.models.clone_model(self.policy_network)
                    self.target_network.trainable = False

            is_checkpoint = save_every and episode % save_every == 0

            if save_path and (is_checkpoint or episode == n_episodes - 1):
                self.save(save_path, episode=episode)

            if not render_interval or steps_done < n_steps:
                self.env.render(
                    episode=episode,
                    max_episodes=n_episodes,
                    max_steps=n_steps
                )  # renderers final state at episode end if not rendered earlier

            self.env.save()

            episode += 1

        mean_reward = total_reward / steps_done

        return mean_reward
