import numpy as np
import tensorflow as tf
from gym.spaces.discrete import Discrete
from tensorflow.keras.losses import MSE
from xagents.base import OffPolicy


class DQN(OffPolicy):
    """
    Playing Atari with Deep Reinforcement Learning.
    https://arxiv.org/abs/1312.5602
    """

    def __init__(
        self,
        envs,
        model,
        buffers,
        double=False,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay_steps=150000,
        target_sync_steps=1000,
        **kwargs,
    ):
        """
        Initialize DQN agent.
        Args:
            envs: A list of gym environments.
            model: tf.keras.models.Model that is expected to be compiled
                with an optimizer before training starts.
            buffers: A list of replay buffer objects whose length should match
                `envs`s'.
            double: If True, DDQN is used.
            epsilon_start: Starting epsilon value which is used to control random exploration.
                It should be decremented and adjusted according to implementation needs.
            epsilon_end: End epsilon value which is the minimum exploration rate.
            epsilon_decay_steps: Number of steps for epsilon to reach `epsilon_end`
                from `epsilon_start`,
            target_sync_steps: Steps to sync target models after each.
            **kwargs: kwargs passed to super classes.
        """
        super(DQN, self).__init__(envs, model, buffers, **kwargs)
        self.assert_valid_env(envs[0], Discrete)
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        self.double = double
        self.epsilon_start = self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_sync_steps = target_sync_steps
        self.batch_indices = tf.range(
            self.buffers[0].batch_size * self.n_envs, dtype=tf.int64
        )[:, tf.newaxis]
        self.batch_dtypes = ['uint8', 'int64', 'float64', 'bool', 'uint8']

    @staticmethod
    def get_action_indices(batch_indices, actions):
        """
        Get indices that will be passed to tf.gather_nd()
        Args:
            batch_indices: tf.range() result of the same shape as the batch size.
            actions: Action tensor of same shape as the batch size.

        Returns:
            Indices as a tensor.
        """
        return tf.concat((batch_indices, tf.cast(actions[:, tf.newaxis], tf.int64)), -1)

    @tf.function
    def get_model_outputs(self, inputs, models, training=True):
        """
        Get model outputs (actions)
        Args:
            inputs: Inputs as tensors / numpy arrays that are expected
                by the given model.
            models: A tf.keras.Model or a list of tf.keras.Model(s)
            training: `training` parameter passed to model call.

        Returns:
            Outputs that is expected from the given model.
        """
        q_values = super(DQN, self).get_model_outputs(inputs, models, training)
        return tf.argmax(q_values, 1), q_values

    def update_epsilon(self):
        """
        Decrement epsilon which aims to gradually reduce randomization.

        Returns:
            None
        """
        self.epsilon = max(
            self.epsilon_end, self.epsilon_start - self.steps / self.epsilon_decay_steps
        )

    def sync_target_model(self):
        """
        Sync target model weights with main's

        Returns:
            None
        """
        if self.steps % self.target_sync_steps == 0:
            self.target_model.set_weights(self.model.get_weights())

    def get_actions(self):
        """
        Generate action following an epsilon-greedy policy.

        Returns:
            A random action or Q argmax.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions, self.n_envs)
        return self.get_model_outputs(self.get_states(), self.output_models)[0]

    def get_targets(self, states, actions, rewards, dones, new_states):
        """
        Get targets for gradient update.
        Args:
            states: A tensor of shape (self.n_envs * total buffer batch size, *self.input_shape)
            actions: A tensor of shape (self.n_envs * total buffer batch size)
            rewards: A tensor of shape (self.n_envs * total buffer batch size)
            dones: A tensor of shape (self.n_envs * total buffer batch size)
            new_states: A tensor of shape (self.n_envs * total buffer batch size, *self.input_shape)

        Returns:
            Target values, a tensor of shape (self.n_envs * total buffer batch size, self.n_actions)
        """
        q_states = self.get_model_outputs(states, self.model)[1]
        if self.double:
            new_state_actions = self.get_model_outputs(new_states, self.model)[0]
            new_state_q_values = self.get_model_outputs(new_states, self.target_model)[
                1
            ]
            a = self.get_action_indices(self.batch_indices, new_state_actions)
            new_state_values = tf.gather_nd(new_state_q_values, a)
        else:
            new_state_values = tf.reduce_max(
                self.get_model_outputs(new_states, self.target_model)[1], axis=1
            )
        new_state_values = tf.where(
            tf.cast(dones, tf.bool),
            tf.constant(0, new_state_values.dtype),
            new_state_values,
        )
        target_values = tf.identity(q_states)
        target_value_update = new_state_values * self.gamma + tf.cast(
            rewards, tf.float32
        )
        indices = self.get_action_indices(self.batch_indices, actions)
        target_values = tf.tensor_scatter_nd_update(
            target_values, indices, target_value_update
        )
        return target_values

    def update_gradients(self, x, y):
        """
        Train on a given batch.
        Args:
            x: States tensor
            y: Targets tensor

        Returns:
            None
        """
        with tf.GradientTape() as tape:
            y_pred = self.get_model_outputs(x, self.model)[1]
            loss = MSE(y, y_pred)
        self.model.optimizer.minimize(loss, self.model.trainable_variables, tape=tape)

    def at_step_start(self):
        """
        Execute steps that will run before self.train_step() which decays epsilon.

        Returns:
            None
        """
        self.update_epsilon()

    @tf.function
    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.envs, batching and gradient updates.

        Returns:
            None
        """
        actions = tf.numpy_function(self.get_actions, [], tf.int64)
        tf.numpy_function(self.step_envs, [actions, False, True], [])
        training_batch = tf.numpy_function(
            self.concat_buffer_samples,
            [],
            self.batch_dtypes,
        )
        targets = self.get_targets(*training_batch)
        self.update_gradients(training_batch[0], targets)

    def at_step_end(self):
        """
        Execute steps that will run after self.train_step() which
        updates target model.

        Returns:
            None
        """
        self.sync_target_model()
