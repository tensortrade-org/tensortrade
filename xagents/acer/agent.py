import numpy as np
import tensorflow as tf
from gym.spaces import Discrete
from xagents import A2C
from xagents.utils.common import LazyFrames


class ACER(A2C):
    """
    Sample Efficient Actor-Critic with Experience Replay.
    https://arxiv.org/abs/1611.01224
    """

    def __init__(
        self,
        envs,
        model,
        buffers,
        ema_alpha=0.99,
        replay_ratio=4,
        epsilon=1e-6,
        importance_c=10.0,
        delta=1,
        trust_region=True,
        **kwargs,
    ):
        """
        Initialize ACER agent.
        Args:
            envs: A list of gym environments.
            model: tf.keras.models.Model that is expected to be compiled
                with an optimizer before training starts.
            buffers: A list of replay buffer objects whose length should match
                `envs`s'.
            ema_alpha: Moving average decay passed to tf.train.ExponentialMovingAverage()
            replay_ratio: Lam value passed to np.random.poisson()
            epsilon: epsilon value used in several calculations during gradient update.
            importance_c: Importance weight truncation parameter.
            delta: Delta parameter used for trust region update.
            trust_region: If False, no trust region updates will be used.
            **kwargs: kwargs Passed to super classes.
        """
        super(ACER, self).__init__(envs, model, **kwargs)
        self.assert_valid_env(envs[0], Discrete)
        self.avg_model = tf.keras.models.clone_model(self.model)
        self.ema = tf.train.ExponentialMovingAverage(ema_alpha)
        self.buffers = buffers
        assert (
            buffers[0].batch_size == 1
        ), f'Buffer batch size should be 1 for ACER, got {buffers[0].batch_size}'
        self.buffer_current_size = tf.Variable(0)
        self.batch_indices = tf.range(self.n_steps * self.n_envs, dtype=tf.int64)[
            :, tf.newaxis
        ]
        self.replay_ratio = replay_ratio
        self.epsilon = epsilon
        self.importance_c = importance_c
        self.delta = delta
        self.trust_region = trust_region
        self.batch_dtypes = ['uint8', 'float32', 'int32', 'float32', 'float32']
        self.batch_shapes = [
            (self.n_envs * (self.n_steps + 1), *self.input_shape),
            (self.n_envs * self.n_steps),
            (self.n_envs * self.n_steps),
            (self.n_envs * self.n_steps),
            (self.n_envs * self.n_steps, self.n_actions),
        ]

    def flat_to_steps(self, t, steps=None):
        """
        Split a tensor to tensors of shape (self.n_envs, self.n_steps)
        Args:
            t: A flat tensor that has (self.n_envs * self.n_steps) values.
            steps: If not specified, self.n_steps is used by default.

        Returns:
            A list of self.n_envs x self.n_steps tensors.
        """
        t = tf.reshape(t, (self.n_envs, steps or self.n_steps, *t.shape[1:]))
        return [
            tf.squeeze(step_t, 1) for step_t in tf.split(t, steps or self.n_steps, 1)
        ]

    def clip_last_step(self, t):
        """
        Remove last step from a given tensor.
        Args:
            t: Tensor that has self.n_steps + 1 items.

        Returns:
            Tensor that has self.n_steps items.
        """
        ts = self.flat_to_steps(t, self.n_steps + 1)
        return tf.reshape(tf.stack(ts[:-1], 1), (-1, *ts[0].shape[1:]))

    @staticmethod
    def add_grads(g1, g2):
        """
        Add 2 gradients if neither is None, otherwise return one of them.
        Args:
            g1: Grad 1
            g2: Grad 2

        Returns:
            One of them or their sum.
        """
        assert g1 is not None or g2 is not None, 'Gradients are None'
        if g1 is not None and g2 is not None:
            return g1 + g2
        if g1 is not None:
            return g1
        return g2

    def update_avg_weights(self):
        """
        Update average model weights after performing gradient update.

        Returns:
            None
        """
        avg_variables = [
            self.ema.average(weight).numpy()
            for weight in self.model.trainable_variables
        ]
        self.avg_model.set_weights(avg_variables)

    def store_batch(self, batch):
        """
        Store batch in self.buffers.
        Args:
            batch: A list of (states, rewards, actions, dones, actor output)

        Returns:
            None
        """
        for i in range(self.n_envs):
            env_outputs = []
            for item in batch:
                current = item[i]
                if len(current.shape) > 2:
                    current = LazyFrames(current.astype(np.uint8))
                env_outputs.append(current)
            self.buffers[i].append(*env_outputs)

    def get_batch(self):
        """
        Get a batch of (states, rewards, actions, dones, actor output),
        save batch to replay buffers and adjust shapes for gradient update.

        Returns:
            Merged environment outputs.
        """
        (
            states,
            rewards,
            actions,
            _,
            dones,
            *_,
            actor_logits,
        ) = super(ACER, self).get_batch()
        states.append(self.get_states())
        batch = [states, rewards, actions, dones[1:], actor_logits]
        batch = [np.asarray(item).swapaxes(0, 1) for item in batch]
        self.store_batch(batch)
        return [
            item.reshape(-1, *item.shape[2:]).astype(dtype)
            for (item, dtype) in zip(batch, self.batch_dtypes)
        ]

    def calculate_returns(
        self,
        rewards,
        dones,
        values=None,
        selected_critic_logits=None,
        selected_importance=None,
    ):
        """
        Get a batch of returns.
        Args:
            rewards: Rewards tensor of shape (self.n_steps, self.n_envs)
            dones: Dones tensor of shape (self.n_steps, self.n_envs)
            values: Values tensor of shape (self.n_steps + 1, self.n_envs).
                required for PPO, TRPO and ACER
            selected_critic_logits: Critic output respective to selected actions
                of shape (self.n_steps, self.n_envs).
                Required for ACER.
            selected_importance: Importance weights respective to selected
                actions of shape (self.n_steps, self.n_envs).
                Required for ACER
        Returns:
            Tensor of n-step returns.
        """
        importance_bar = self.flat_to_steps(tf.minimum(1.0, selected_importance))
        dones = self.flat_to_steps(dones)
        rewards = self.flat_to_steps(rewards)
        selected_critic_logits = self.flat_to_steps(selected_critic_logits)
        values = self.flat_to_steps(values, self.n_steps + 1)
        current_return = values[-1]
        returns = []
        for i in reversed(range(self.n_steps)):
            current_return = rewards[i] + self.gamma * current_return * (1.0 - dones[i])
            returns.append(current_return)
            current_return = (
                importance_bar[i] * (current_return - selected_critic_logits[i])
            ) + values[i]
        return tf.reshape(tf.stack(returns[::-1], 1), [-1])

    def calculate_losses(
        self,
        action_probs,
        values,
        returns,
        selected_probs,
        selected_importance,
        selected_critic_logits,
    ):
        """
        Calculate model loss.
        Args:
            action_probs: A tensor of shape (self.n_envs * self.n_steps, self.n_actions)
            values: A tensor of shape self.n_envs * (self.n_steps + 1)
            returns: A tensor of shape (self.n_envs * self.n_steps)
            selected_probs: A tensor of shape (self.n_envs * self.n_steps)
            selected_importance: A tensor of shape (self.n_envs * self.n_steps)
            selected_critic_logits: A tensor of shape (self.n_envs * self.n_steps)

        Returns:
            Loss as a TF tensor if trust region is not used, otherwise loss, value_loss.
        """
        entropy = tf.reduce_mean(
            -tf.reduce_sum(
                action_probs * tf.math.log(action_probs + self.epsilon), axis=1
            )
        )
        values = self.clip_last_step(values)
        advantages = returns - values
        log_probs = tf.math.log(selected_probs + self.epsilon)
        action_gain = log_probs * tf.stop_gradient(
            advantages * tf.minimum(self.importance_c, selected_importance)
        )
        action_loss = -tf.reduce_mean(action_gain)
        value_loss = (
            tf.reduce_mean(
                tf.square(tf.stop_gradient(returns) - selected_critic_logits) * 0.5
            )
            * self.value_loss_coef
        )
        if self.trust_region:
            return (
                -(action_loss - self.entropy_coef * entropy)
                * self.n_steps
                * self.n_envs
            ), value_loss
        return (
            action_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy
        )

    def calculate_grads(self, tape, losses, action_probs, avg_action_probs):
        """
        Calculate gradients given loss(es)
        Args:
            tape: tf.GradientTape()
            losses: loss or loss, value_loss (if trust region is used)
            action_probs: A tensor of shape (self.n_envs * self.n_steps, self.n_actions)
            avg_action_probs:  A tensor of shape (self.n_envs * self.n_steps, self.n_actions)

        Returns:
            A list of gradients.
        """
        if not self.trust_region:
            return tape.gradient(losses, self.model.trainable_variables)
        loss, value_loss = losses
        g = tape.gradient(
            loss,
            action_probs,
        )
        k = -avg_action_probs / (action_probs + self.epsilon)
        adj = tf.maximum(
            0.0,
            (tf.reduce_sum(k * g, axis=-1) - self.delta)
            / (tf.reduce_sum(tf.square(k), axis=-1) + self.epsilon),
        )
        g = g - tf.reshape(adj, [self.n_envs * self.n_steps, 1]) * k
        output_grads = -g / (self.n_envs * self.n_steps)
        action_grads = tape.gradient(
            action_probs, self.model.trainable_variables, output_grads
        )
        value_grads = tape.gradient(value_loss, self.model.trainable_variables)
        return [self.add_grads(g1, g2) for (g1, g2) in zip(action_grads, value_grads)]

    def update_gradients(
        self,
        states,
        rewards,
        actions,
        dones,
        previous_action_probs,
    ):
        """
        Perform gradient updates.
        Args:
            states: A tensor of shape (self.n_envs * self.n_steps, *self.input_shape)
            rewards: A tensor of shape (self.n_envs * self.n_steps)
            actions: A tensor of shape (self.n_envs * self.n_steps)
            dones: A tensor of shape (self.n_envs * self.n_steps)
            previous_action_probs: A tensor of shape (self.n_envs * self.n_steps, self.n_actions)

        Returns:
            None
        """
        action_indices = tf.concat(
            (self.batch_indices, tf.cast(actions[:, tf.newaxis], tf.int64)), -1
        )
        with tf.GradientTape(True) as tape:
            *_, critic_logits, _, action_probs = self.get_model_outputs(
                states, self.model
            )
            *_, avg_action_probs = self.get_model_outputs(states, self.avg_model)
            values = tf.reduce_sum(action_probs * critic_logits, axis=-1)
            action_probs = self.clip_last_step(action_probs)
            avg_action_probs = self.clip_last_step(avg_action_probs)
            critic_logits = self.clip_last_step(critic_logits)
            selected_probs = tf.gather_nd(action_probs, action_indices)
            selected_critic_logits = tf.gather_nd(critic_logits, action_indices)
            importance_weights = action_probs / (previous_action_probs + self.epsilon)
            selected_importance = tf.gather_nd(importance_weights, action_indices)
            returns = self.calculate_returns(
                rewards, dones, values, selected_critic_logits, selected_importance
            )
            losses = self.calculate_losses(
                action_probs,
                values,
                returns,
                selected_probs,
                selected_importance,
                selected_critic_logits,
            )
        grads = self.calculate_grads(tape, losses, action_probs, avg_action_probs)
        if self.grad_norm is not None:
            grads, norm_grads = tf.clip_by_global_norm(grads, self.grad_norm)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.ema.apply(self.model.trainable_variables)
        tf.numpy_function(self.update_avg_weights, [], [])

    def set_batch_shapes(self, batch):
        """
        Set unknown shape batch items to their appropriate shapes,
        which for some reason, tf doesn't keep track of.
        Args:
            batch: A list of unknown shape tensors which shapes are expected
                to match self.batch_shapes.

        Returns:
            None
        """
        for item, shape in zip(batch, self.batch_shapes):
            item.set_shape(shape)

    @tf.function
    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.envs, batching and gradient updates.

        Returns:
            None
        """
        batch = tf.numpy_function(self.get_batch, [], self.batch_dtypes)
        self.buffer_current_size.assign_add(1)
        self.set_batch_shapes(batch)
        self.update_gradients(*batch)
        if (
            self.replay_ratio > 0
            and self.buffer_current_size >= self.buffers[0].initial_size
        ):
            for _ in range(np.random.poisson(self.replay_ratio)):
                batch = tf.numpy_function(
                    self.concat_buffer_samples,
                    [],
                    self.batch_dtypes,
                )
                self.set_batch_shapes(batch)
                self.update_gradients(*batch)
