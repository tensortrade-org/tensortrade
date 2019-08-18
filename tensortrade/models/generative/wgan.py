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
#
# Modified source: https://github.com/timsainb/tensorflow2-generative-models/blob/master/3.0-WGAN-GP-fashion-mnist.ipynb
# Source reference: https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2/
# Original paper: https://arxiv.org/abs/1701.07875

import tensorflow as tf


class WGAN(tf.keras.Model):
    def __init__(self, generator: tf.keras.Sequential, discriminator: tf.keras.Sequential,  **kwargs):
        super().__init__()

        self.n_samples = kwargs.get('n_samples', 64)
        self.gradient_penalty_weight = kwargs.get('gradient_penalty_weight', 10.0)

        self.generator_lr = kwargs.get('generator_lr', 0.0001)
        self.generator_beta_1 = kwargs.get('generator_beta_1', 0.5)
        self.discriminator_lr = kwargs.get('discriminator_lr', 0.0005)

        self.generator = generator
        self.discriminator = discriminator

        self.gen_optimizer = tf.keras.optimizers.Adam(
            self.generator_lr, beta_1=self.generator_beta_1)
        self.disc_optimizer = tf.keras.optimizers.RMSprop(self.discriminator_lr)

    def generate(self, z):
        return self.generator(z)

    def discriminate(self, x):
        return self.discriminator(x)

    def generate_random(self):
        return self.generate(tf.random.normal(shape=(1, self.n_samples)))

    def gradient_penalty(self, x, x_gen):
        epsilon = tf.random.uniform([x.shape[0], 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * x + (1 - epsilon) * x_gen

        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.discriminate(x_hat)

        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)

        return d_regularizer

    def compute_loss(self, x):
        z_samp = tf.random.normal([x.shape[0], 1, 1, self.n_samples])

        x_gen = self.generate(z_samp)
        logits_x = self.discriminate(x)
        logits_x_gen = self.discriminate(x_gen)

        d_regularizer = self.gradient_penalty(x, x_gen)
        disc_loss = (
            tf.reduce_mean(logits_x)
            - tf.reduce_mean(logits_x_gen)
            + d_regularizer * self.gradient_penalty_weight
        )

        gen_loss = tf.reduce_mean(logits_x_gen)

        return disc_loss, gen_loss

    def compute_gradients(self, x):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            disc_loss, gen_loss = self.compute_loss(x)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        return gen_gradients, disc_gradients

    def apply_gradient(self, model, optimizer, gradients):
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    @tf.function
    def train(self, train_x):
        gen_gradients, disc_gradients = self.compute_gradients(train_x)

        self.apply_gradient(self.generator, self.gen_optimizer, gen_gradients)
        self.apply_gradient(self.discriminator, self.disc_optimizer, disc_gradients)
