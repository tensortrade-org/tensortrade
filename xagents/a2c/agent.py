import numpy as np
import tensorflow as tf
from gym.spaces.discrete import Discrete
from tensorflow_probability.python.distributions import (
    Categorical, MultivariateNormalDiag)
from xagents.base import OnPolicy


class A2C(OnPolicy):
    """
    Asynchronous Methods for Deep Reinforcement Learning
    https://arxiv.org/abs/1602.01783
    """

    def __init__(
        self,
        envs,
        model,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        grad_norm=0.5,
        **kwargs,
    ):
        """
        Initialize A2C agent.
        Args:
            envs: A list of gym environments.
            model: tf.keras.models.Model that is expected to be compiled
                with an optimizer before training starts.
            entropy_coef: Entropy coefficient used for entropy loss calculation.
            value_loss_coef: Value coefficient used for value loss calculation.
            grad_norm: Gradient clipping value passed to tf.clip_by_global_norm()
            **kwargs: kwargs Passed to super classes.
        """
        super(A2C, self).__init__(envs, model, **kwargs)
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.grad_norm = grad_norm
        assert (
            len(model.layers) > 2
        ), f'Expected a model that has at least 3 layers, got {len(model.layers)}'
        activations = [layer.activation for layer in model.layers[-2:]]
        self.output_is_softmax = tf.keras.activations.softmax in activations
        self.distribution_type = (
            Categorical
            if isinstance(self.envs[0].action_space, Discrete)
            else MultivariateNormalDiag
        )

    def get_distribution(self, actor_output):
        """
        Get a probability distribution from probabilities or logits.
        Args:
            actor_output: Output by the actor model (probabilities/logits).

        Returns:
            tfp.python.distributions.Categorical
        """
        if self.distribution_type == MultivariateNormalDiag:
            return MultivariateNormalDiag(actor_output)
        if self.output_is_softmax:
            return Categorical(probs=actor_output)
        return Categorical(logits=actor_output)

    def get_model_outputs(self, inputs, models, training=True, actions=None):
        """
        Get actor and critic outputs, and determine actions sampled from
        respective probability distribution.
        Args:
            inputs: Inputs as tensors / numpy arrays that are expected
                by the given model(s).
            models: A tf.keras.Model or a list of tf.keras.Model(s)
            training: `training` parameter passed to model call.
            actions: If not specified, a sample is drawn from corresponding
                distribution and used for calculation of log probs.

        Returns:
            [actions, log probs, critic output, entropy, actor output]
        """
        actor_output, critic_output = super(A2C, self).get_model_outputs(
            inputs, models, training
        )
        distribution = self.get_distribution(actor_output)
        critic_output = tf.squeeze(critic_output)
        if actions is None:
            actions = distribution.sample(seed=self.seed)
        action_log_probs = distribution.log_prob(actions)
        return (
            actions,
            action_log_probs,
            critic_output,
            distribution.entropy(),
            actor_output,
        )

    def get_batch(self):
        """
        Get n-step batch which is the result of running self.envs step() for
        self.n_steps times.

        Returns:
            A list of numpy arrays which contains
             [states, rewards, actions, critic_output, dones, log probs, entropies, actor_output]
        """
        batch = (
            states,
            rewards,
            actions,
            critic_output,
            dones,
            log_probs,
            entropies,
            actor_output,
        ) = [[] for _ in range(8)]
        step_states = tf.numpy_function(self.get_states, [], tf.float32)
        step_dones = tf.numpy_function(self.get_dones, [], tf.float32)
        for _ in range(self.n_steps):
            (
                step_actions,
                step_log_probs,
                step_values,
                step_entropies,
                step_actor_logits,
            ) = self.get_model_outputs(step_states, self.output_models)
            states.append(step_states)
            actions.append(step_actions)
            critic_output.append(step_values)
            log_probs.append(step_log_probs)
            dones.append(step_dones)
            entropies.append(step_entropies)
            actor_output.append(step_actor_logits)
            *_, step_rewards, step_dones, step_states = tf.numpy_function(
                self.step_envs,
                [step_actions, True, False],
                5 * [tf.float32],
            )
            rewards.append(step_rewards)
        dones.append(step_dones)
        return batch

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
        next_values = self.get_model_outputs(self.get_states(), self.output_models)[2]
        returns = [next_values]
        for step in reversed(range(self.n_steps)):
            returns.append(
                rewards[step] + self.gamma * returns[-1] * (1.0 - dones[step + 1])
            )
        return np.asarray(returns[::-1], np.float32)[:-1]

    def np_train_step(self):
        """
        Perform the batching and return calculation in numpy.
        """
        (
            states,
            rewards,
            actions,
            critic_output,
            dones,
            log_probs,
            entropies,
            actor_output,
        ) = [np.asarray(item, np.float32) for item in self.get_batch()]
        returns = self.calculate_returns(rewards, dones)
        return self.concat_step_batches(states, returns, actions, critic_output)

    @tf.function
    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.envs, batching and gradient updates.

        Returns:
            None
        """
        states, returns, actions, old_values = tf.numpy_function(
            self.np_train_step, [], 4 * [tf.float32]
        )
        advantages = returns - old_values
        with tf.GradientTape() as tape:
            _, log_probs, critic_output, entropy, actor_output = self.get_model_outputs(
                states, self.output_models, actions=actions
            )
            entropy = tf.reduce_mean(entropy)
            pg_loss = -tf.reduce_mean(advantages * log_probs)
            value_loss = tf.reduce_mean(tf.square(critic_output - returns))
            loss = (
                pg_loss
                - entropy * self.entropy_coef
                + value_loss * self.value_loss_coef
            )
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_norm)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
