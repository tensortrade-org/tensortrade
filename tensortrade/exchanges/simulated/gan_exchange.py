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
import numpy as np
import pandas as pd

from gym import spaces
from typing import Dict

from tensortrade.trades import Trade, TradeType
from tensortrade.slippage import RandomUniformSlippageModel
from tensortrade.exchanges import InstrumentExchange


class GANExchange(InstrumentExchange):
    """A simulated instrument exchange, in which the price history is based off a generative adversarial network
    model with supplied parameters.

    If the `training_data` parameter is not supplied upon initialization, it must be set before
    the exchange can be used within a trading environment.
    """

    def __init__(self, training_data: pd.DataFrame = None, **kwargs):
        super().__init__(**kwargs)

        if training_data is not None:
            self._training_data = training_data

        self._prices_per_gen = kwargs.get('prices_per_gen', 1000)
        self._n_samples = kwargs.get('n_samples', 64)
        self._output_shape = kwargs.get('output_shape', (self._prices_per_gen, 5, 1))

        self._initialize_gan()

    def _initialize_gan(self):
        generator = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1, self._n_samples)),
            tf.keras.layers.Dense(units=(self._prices_per_gen + 3) *
                                  8 * self._n_samples, activation="relu"),
            tf.keras.layers.Reshape(target_shape=((self._prices_per_gen + 3), 8, self._n_samples)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu"
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu"
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
            ),
            tf.keras.layers.Reshape(target_shape=(self._output_shape))
        ])

        discriminator = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self._output_shape),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=1, activation="sigmoid"),
        ])

        self._gan = {'generator': generator,
                     'discriminator': discriminator}

    def reset(self):
        super().reset()
