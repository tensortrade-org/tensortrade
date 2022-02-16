import tensorflow as tf
from gym.spaces import Box
from tensorflow.keras.losses import MSE
from xagents.base import OffPolicy


class DDPG(OffPolicy):
    """
    Continuous control with deep reinforcement learning.
    https://arxiv.org/abs/1509.02971
    """

    def __init__(
        self,
        envs,
        actor_model,
        critic_model,
        buffers,
        gradient_steps=None,
        tau=0.05,
        step_noise_coef=0.1,
        **kwargs,
    ):
        """
        Initialize DDPG agent.
        Args:
            envs: A list of gym environments.
            actor_model: Actor separate model as a tf.keras.Model.
            critic_model: Critic separate model as a tf.keras.Model.
            buffers: A list of replay buffer objects whose length should match
                `envs`s'.
            gradient_steps: Number of iterations per train_step() call, if not
                specified, it defaults to the number of steps per most-recent
                finished episode per environment.
            tau: Tau constant used for syncing target model weights.
            policy_noise_coef: Coefficient multiplied by noise added to target actions.
            **kwargs: kwargs passed to super classes.
        """
        super(DDPG, self).__init__(envs, actor_model, buffers, **kwargs)
        self.assert_valid_env(envs[0], Box)
        self.actor = actor_model
        self.critic = critic_model
        self.policy_delay = 1
        self.gradient_steps = gradient_steps
        self.tau = tau
        self.step_noise_coef = step_noise_coef
        self.episode_steps = tf.Variable(tf.zeros(self.n_envs), False)
        self.step_increment = tf.ones(self.n_envs)
        self.output_models.append(self.critic)
        self.target_actor = tf.keras.models.clone_model(self.actor)
        self.target_critic = tf.keras.models.clone_model(self.critic)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        self.model_groups = [
            (self.actor, self.target_actor),
            (self.critic, self.target_critic),
        ]
        self.batch_dtypes = 5 * ['float32']

    def get_step_actions(self):
        """
        Get self.n_envs noisy actions to be stepped by self.envs

        Returns:
            actions
        """
        actions = self.actor(tf.numpy_function(self.get_states, [], tf.float64))
        actions += tf.random.normal(
            shape=(self.n_envs, self.n_actions), stddev=self.step_noise_coef
        )
        return tf.clip_by_value(actions, -1, 1)

    def sync_target_models(self):
        """
        Sync target actor, target critic 1, and target critic 2 weights
        with their respective models in self.model_groups.

        Returns:
            None
        """
        for model, target_model in self.model_groups:
            for var, target_var in zip(
                model.trainable_variables, target_model.trainable_variables
            ):
                target_var.assign((1 - self.tau) * target_var + self.tau * var)

    def update_actor_weights(self, states):
        """
        Update actor weights.
        Args:
            states: A tensor of shape (self.n_envs * total buffer batch size, *self.input_shape)

        Returns:
            None.
        """
        with tf.GradientTape() as tape:
            actor_loss = -tf.reduce_mean(
                self.critic(tf.concat([states, self.actor(states)], 1))
            )
        self.actor.optimizer.minimize(
            actor_loss, self.actor.trainable_variables, tape=tape
        )

    def update_critic_weights(self, states, actions, new_states, dones, rewards):
        """
        Update critic weights.
        Args:
            states: A tensor of shape (self.n_envs * total buffer batch size, *self.input_shape)
            actions: A tensor of shape (self.n_envs * total buffer batch size, self.n_actions)
            new_states: A tensor of shape (self.n_envs * total buffer batch size, *self.input_shape)
            dones: A tensor of shape (self.n_envs * total buffer batch size)
            rewards: A tensor of shape (self.n_envs * total buffer batch size)

        Returns:
            None
        """
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_states)
            target_critic_input = tf.concat([new_states, target_actions], 1)
            target_value = self.target_critic(target_critic_input)
            target_value = rewards + tf.stop_gradient(
                (1 - dones) * self.gamma * target_value
            )
            critic_input = tf.concat([states, actions], 1)
            value = self.critic(critic_input)
            loss = MSE(value, target_value)
        self.critic.optimizer.minimize(loss, self.critic.trainable_variables, tape=tape)

    def update_weights(self, gradient_steps):
        """
        Run gradient steps and update both actor and critic weights according
            to self.policy delay for the given gradient steps.
        Args:
            gradient_steps: Number of iterations.

        Returns:
            None.
        """
        for gradient_step in range(int(gradient_steps)):
            states, actions, rewards, dones, new_states = tf.numpy_function(
                self.concat_buffer_samples, [], self.batch_dtypes
            )
            self.update_critic_weights(states, actions, new_states, dones, rewards)
            if gradient_step % self.policy_delay == 0:
                self.update_actor_weights(states)
                self.sync_target_models()

    @tf.function
    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.envs, batching and gradient updates.

        Returns:
            None
        """
        step_actions = self.get_step_actions()
        *_, dones, _ = tf.numpy_function(
            self.step_envs, [step_actions, True, True], self.batch_dtypes
        )
        for done_idx in tf.where(dones):
            gradient_steps = self.gradient_steps or self.episode_steps[done_idx[0]]
            self.update_weights(gradient_steps)
        self.episode_steps.assign(
            (self.episode_steps + self.step_increment) * (1 - dones)
        )
