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

import pandas as pd

from time import time

import numpy as np
import pandas as pd
import tensorflow as tf

from stochastic.noise import GaussianNoise


class GAN:

    def __init__(self, training_data: pd.DataFrame = None, **kwargs):
        self._training_data = training_data
        self._n_samples = kwargs.get('n_samples', 5000)
        self._base_price = kwargs.get('base_price', 1)
        self._base_volume = kwargs.get('base_volume', 1)
        self._start_date = kwargs.get('start_date', '2010-01-01')
        self._start_date_format = kwargs.get('start_date_format', '%Y-%m-%d')
        self._times_to_generate = kwargs.get('times_to_generate', self._n_samples)
        self._timeframe = kwargs.get('timeframe', '1H')
        self._cagr = kwargs.get('cagr', 0)
        self._labels = kwargs.get('labels', None)
        self._vis_freq = kwargs.get('vis_freq', 2)
        self._volatility = kwargs.get('volatility', 0)
        self._t = kwargs.get('trading_days', 9)
        self._d_rounds = kwargs.get('D_rounds', 3)
        self._g_rounds = kwargs.get('G_rounds', 1)
        self._lr = kwargs.get('learning_rate', 0.001)
        self._batch_size = kwargs.get('batch_size', 100)
        self._latent_dim = kwargs.get('latent_dim', 20)
        self._num_epochs = kwargs.get('num_epochs', 20)
        self._seq_length = kwargs.get('seq_length', 10)
        self._hidden_units_g = kwargs.get('hidden_units_g', 150)
        self._hidden_units_d = kwargs.get('hidden_units_d', 150)
        self._num_generated_features = kwargs.get('num_generated_features', 1)
        self._output_shape = kwargs.get('output_shape', (self._times_to_generate, 5, 1))
        self._initialize_gan()
        self._train_gan()
        self.generate_price_history()

    def _initialize_placeholders(self):
        self._CG = tf.placeholder(tf.float32,
                                  [self._batch_size, self._seq_length])
        self._CD = tf.placeholder(tf.float32, [self._batch_size, self._seq_length])
        self._Z = tf.placeholder(tf.float32, [self._batch_size, self._seq_length, self._latent_dim])
        self._W_out_G = tf.Variable(tf.truncated_normal([self._hidden_units_g, self._num_generated_features]))
        self._b_out_G = tf.Variable(tf.truncated_normal([self._num_generated_features]))

        self._X = tf.placeholder(tf.float32, [self._batch_size, self._seq_length, self._num_generated_features])
        self._W_out_D = tf.Variable(tf.truncated_normal([self._hidden_units_d, 1]))
        self._b_out_D = tf.Variable(tf.truncated_normal([1]))

    @staticmethod
    def sample_Z(batch_size, seq_length, latent_dim):
        # --- to do with latent space --- #
        sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))
        return sample

    def _cal_cagr_n_vol(self):
        days = (self._training_data.index[-1] - self._training_data.index[0]).days
        self._cagr = (((self._training_data['Close'][-1]) / self._training_data['Close'][1]) ** (252.0 / days)) - 1

        # create a series of percentage returns and calculate the annual volatility of returns
        self._training_data['Returns'] = self._training_data['Close'].pct_change()
        self._volatility = self._training_data['Returns'].std() * np.sqrt(self._t)

    def sample_data(self):
        """
        we will do this by usual Monte Carlo simulation to look at the potential evolution of asset prices
        over time, assuming they are subject to daily returns that follow a normal distribution.
        To set up our simulation, we need to estimate the expected level of return (mu) and volatility (vol) of the stock
        """
        vectors = []
        for i in range(self._n_samples):
            # create list of daily returns using random normal distribution
            daily_returns = np.random.normal(self._cagr / self._t, self._volatility / np.sqrt(self._t),
                                             self._seq_length)
            vectors.append(np.log(daily_returns + np.absolute(min(daily_returns)) + 1))

        dataset = np.array(vectors)
        dataset.reshape(-1, self._seq_length)

        return dataset

    @staticmethod
    def get_batch(samples, batch_size, batch_idx):
        start_pos = batch_idx * batch_size
        end_pos = start_pos + batch_size
        return samples[start_pos:end_pos]

    def generator(self, z, c):
        with tf.variable_scope("generator") as scope:
            # each step of the generator takes a random seed + the conditional embedding
            repeated_encoding = tf.tile(c, [1, tf.shape(z)[1]])
            repeated_encoding = tf.reshape(repeated_encoding, [tf.shape(z)[0], tf.shape(z)[1], self._seq_length])
            generator_input = tf.concat([repeated_encoding, z], 2)

            cell = tf.contrib.rnn.LSTMCell(num_units=self._hidden_units_g, state_is_tuple=True)
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float32,
                sequence_length=[self._seq_length] * self._batch_size,
                inputs=generator_input)
            rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, self._hidden_units_g])
            logits_2d = tf.matmul(rnn_outputs_2d, self._W_out_G) + self._b_out_G
            output_2d = tf.nn.tanh(logits_2d)
            output_3d = tf.reshape(output_2d, [-1, self._seq_length, self._num_generated_features])
        return output_3d

    def discriminator(self, x, c, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            repeated_encoding = tf.tile(c, [1, tf.shape(x)[1]])
            repeated_encoding = tf.reshape(repeated_encoding, [tf.shape(x)[0], tf.shape(x)[1], self._seq_length])
            decoder_input = tf.concat([repeated_encoding, x], 2)

            cell = tf.contrib.rnn.LSTMCell(num_units=self._hidden_units_d, state_is_tuple=True)
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float32,
                inputs=decoder_input)
            rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, self._hidden_units_g])
            logits = tf.matmul(rnn_outputs_flat, self._W_out_D) + self._b_out_D
            output = tf.nn.sigmoid(logits)
        return output, logits

    def price_gen(self, ind_gen_sample=1):
        daily_log_returns = self._generated_data[0, :, ind_gen_sample]

        price_list = [self._training_data['Close'][-1]]  # initial price (today)

        for x in np.transpose(daily_log_returns):
            price_list.append(price_list[-1] * np.exp(x))

        return price_list

    def price_real(self):
        daily_returns = np.random.normal(self._cagr / self._t, self._volatility / np.sqrt(self._t), self._t) + 1

        price_list = [self._training_data['Close'][-1]]

        for x in daily_returns:
            price_list.append(price_list[-1] * x)

        return price_list

    def prices_gen_data_frame(self):
        df = pd.DataFrame([])

        for i in range(self._n_samples):
            df[i] = self.price_real()

        return df

    def prices_real_data_frame(self):
        df = pd.DataFrame()

        for i in range(self._n_samples):
            df[i] = self.price_real()

        return df

    def train_generator(self, batch_idx, offset):
        # update the generator
        for g in range(self._g_rounds):
            Y_mb = self.get_batch(self._samples, self._batch_size, batch_idx + g + offset)
            _, G_loss_curr = self._sess.run([self._G_solver, self._G_loss],
                                            feed_dict={self._CG: Y_mb,
                                                       self._Z: self.sample_Z(self._batch_size, self._seq_length,
                                                                              self._latent_dim)})
        return G_loss_curr

    def train_discriminator(self, batch_idx, offset):
        # update the discriminator
        for d in range(self._d_rounds):
            # using same input sequence for both the synthetic data and the real one,
            # probably it is not a good idea...
            X_mb = self.get_batch(self._samples, self._batch_size, batch_idx + d + offset)
            X_mb = X_mb.reshape(self._batch_size, self._seq_length, 1)
            Y_mb = self.get_batch(self._samples, self._batch_size, batch_idx + d + offset)
            _, D_loss_curr = self._sess.run([self._D_solver, self._D_loss],
                                            feed_dict={self._CD: Y_mb, self._CG: Y_mb, self._X: X_mb,
                                                       self._Z: self.sample_Z(self._batch_size, self._seq_length,
                                                                              self._latent_dim)})
        return D_loss_curr

    def _initialize_gan(self):
        tf.reset_default_graph()

        self._cal_cagr_n_vol()

        self._samples = self.sample_data()
        self._initialize_placeholders()

        # Define the loss function for the GAN
        self._g_sample = self.generator(self._Z, self._CG)
        d_real, d_logit_real = self.discriminator(self._X, self._CD)
        d_fake, d_logit_fake = self.discriminator(self._g_sample, self._CG, reuse=True)

        generator_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
        discriminator_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]

        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real,
                                                                             labels=tf.ones_like(d_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake,
                                                                             labels=tf.zeros_like(d_logit_fake)))
        self._D_loss = D_loss_real + D_loss_fake
        self._G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake,
                                                                              labels=tf.ones_like(d_logit_fake)))
        # Define the optimisers (GD for the discriminator and Adam for generator)
        self._D_solver = tf.train.GradientDescentOptimizer(learning_rate=self._lr).minimize(self._D_loss,
                                                                                            var_list=discriminator_vars)
        self._G_solver = tf.train.AdamOptimizer().minimize(self._G_loss, var_list=generator_vars)

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self._vis_Z = self.sample_Z(self._batch_size, self._seq_length, self._latent_dim)

    def _train_gan(self):
        d_loss = []
        g_loss = []
        gen_samples = []
        start = time()
        for num_epoch in range(self._num_epochs):

            for batch_idx in range(0, int(len(self._samples) / self._batch_size) - (self._d_rounds + self._g_rounds),
                                   self._d_rounds + self._g_rounds):

                if num_epoch % 2 == 0:

                    G_loss_curr = self.train_generator(batch_idx, 0)
                    D_loss_curr = self.train_discriminator(batch_idx, self._g_rounds)

                else:

                    D_loss_curr = self.train_discriminator(batch_idx, 0)
                    G_loss_curr = self.train_generator(batch_idx, self._d_rounds)

                d_loss.append(D_loss_curr)
                g_loss.append(G_loss_curr)
                t = time() - start

                print(num_epoch, '\t', D_loss_curr, '\t', G_loss_curr, '\t', t)

            # save synthetic data
            if num_epoch % 5 == 0:
                # generate synthetic dataset
                for batch_idx in range(int(len(self._samples) / self._batch_size)):
                    X_mb = self.get_batch(self._samples, self._batch_size, batch_idx)
                    Y_mb = self.get_batch(self._samples, self._batch_size, batch_idx)
                    z_ = self.sample_Z(self._batch_size, self._seq_length, self._latent_dim)
                    gen_samples_mb = self._sess.run(self._g_sample, feed_dict={self._Z: z_, self._CG: Y_mb})
                    gen_samples.append(gen_samples_mb)
                    print(batch_idx)

                self.gen_samples = np.vstack(gen_samples)

        ax = pd.DataFrame(
            {
                'Generative Loss': g_loss,
                'Discriminative Loss': d_loss,
            }
        ).plot(title='Training loss', logy=True)
        ax.set_xlabel("Training iterations")
        ax.set_ylabel("Loss")

        self._generated_data = np.transpose(gen_samples)
        # plot the log-returns
        gen_ind = 1  # change in function price as well
        pd.DataFrame(self._generated_data[0, :,
                     gen_ind]).plot()  # 1 is the index of the plotted sample out of the 1000 generated

    @staticmethod
    def transform2darray(vector):
        v = []
        for i in range(len(vector)):
            v.append(vector[i][0])
        return v

    def generate_price_history(self):
        prices = self.prices_gen_data_frame().iloc[[self._seq_length - 1]]
        prices = self.transform2darray(np.transpose(np.asarray(prices)))
        volume_gen = GaussianNoise(t=self._times_to_generate)
        volumes = volume_gen.sample(self._times_to_generate)

        start_date = pd.to_datetime(self._start_date, format=self._start_date_format)
        price_frame = pd.DataFrame([], columns=['date', 'price'], dtype=float)
        volume_frame = pd.DataFrame([], columns=['date', 'volume'], dtype=float)

        price_frame['date'] = pd.date_range(
            start=start_date, periods=self._times_to_generate, freq="1min")
        print(len(prices), len(price_frame))
        price_frame['price'] = prices

        volume_frame['date'] = price_frame['date'].copy()
        volume_frame['volume'] = abs(volumes)

        price_frame.set_index('date')
        price_frame.index = pd.to_datetime(price_frame.index, unit='m', origin=start_date)

        volume_frame.set_index('date')
        volume_frame.index = pd.to_datetime(volume_frame.index, unit='m', origin=start_date)

        data_frame = price_frame['price'].resample(self._timeframe).ohlc()
        data_frame['volume'] = volume_frame['volume'].resample(self._timeframe).sum()

        return data_frame


def gan(training_data: pd.DataFrame = None,
        base_price: int = 1,
        base_volume: int = 1,
        n_samples: int = 5000,
        start_date: str = '2010-01-01',
        start_date_format: str = '%Y-%m-%d',
        times_to_generate: int = 5000,
        time_frame: str = '1h',
        cagr: int = 0,
        labels = None,
        vis_freq: int = 2,
        volatility: int = 0,
        t: int = 9,
        d_rounds: int = 3,
        g_rounds: int = 1,
        lr: float = 0.001,
        batch_size: int = 100,
        latent_dim: int = 20,
        num_epochs: int = 20,
        seq_length: int = 10,
        hidden_units_g: int = 150,
        hidden_units_d: int = 150,
        num_generated_features: int = 1,
        ):

    options = {
        'base_price': base_price,
        'base_volume': base_volume,
        'n_samples': n_samples,
        'start_date': start_date,
        'start_date_format': start_date_format,
        'times_to_generate': times_to_generate,
        'timeframe': time_frame,
        'cagr': cagr,
        'labels': labels,
        'vis_freq': vis_freq,
        'volatility': volatility,
        't': t,
        'd_rounds': d_rounds,
        'g_rounds': g_rounds,
        'lr': lr,
        'batch_size': batch_size,
        'latent_dim': latent_dim,
        'num_epochs': num_epochs,
        'seq_length': seq_length,
        'hidden_units_g': hidden_units_g,
        'hidden_units_d': hidden_units_d,
        'num_generated_features': num_generated_features,
        'output_shape': tuple((times_to_generate, 5, 1))
    }

    gan_gen = GAN(training_data=training_data, **options)

    return gan_gen.generate_price_history()
