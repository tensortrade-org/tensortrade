import tensorflow as tf
from tensorflow.keras.losses import MSE
from xagents import DDPG


class TD3(DDPG):
    """
    Addressing Function Approximation Error in Actor-Critic Methods.
    https://arxiv.org/abs/1802.09477
    """

    def __init__(
        self,
        envs,
        actor_model,
        critic_model,
        buffers,
        policy_delay=2,
        policy_noise_coef=0.2,
        noise_clip=0.5,
        **kwargs,
    ):
        """
        Initialize TD3 agent
        Args:
            envs: A list of gym environments.
            actor_model: Actor separate model as a tf.keras.Model.
            critic_model: Critic separate model as a tf.keras.Model.
            buffers: A list of replay buffer objects whose length should match
                `envs`s'.
            policy_delay: Number of gradient steps after which, actor weights
                will be updated as well as syncing the target models weights.
            policy_noise_coef: Coefficient multiplied by noise added to target actions.
            noise_clip: Target noise clipping value.
            **kwargs: kwargs passed to super classes.
        """
        super(TD3, self).__init__(envs, actor_model, critic_model, buffers, **kwargs)
        self.critic1 = self.critic
        self.target_critic1 = self.target_critic
        self.policy_delay = policy_delay
        self.policy_noise_coef = policy_noise_coef
        self.noise_clip = noise_clip
        self.critic2 = tf.keras.models.clone_model(self.critic1)
        self.output_models.append(self.critic2)
        self.critic2.compile(
            tf.keras.optimizers.get(self.critic1.optimizer.get_config()['name'])
        )
        self.critic2.optimizer.learning_rate.assign(
            self.critic1.optimizer.learning_rate
        )
        self.target_critic2 = tf.keras.models.clone_model(self.critic1)
        self.target_critic2.set_weights(self.critic2.get_weights())
        self.model_groups.append(
            (self.critic2, self.target_critic2),
        )

    def get_step_actions(self):
        """
        Get self.n_envs actions to be stepped by self.envs

        Returns:
            actions
        """
        return self.actor(tf.numpy_function(self.get_states, [], self.states[0].dtype))

    def update_critic_weights(self, states, actions, new_states, dones, rewards):
        """
        Update critic 1 and critic 2 weights.
        Args:
            states: A tensor of shape (self.n_envs * total buffer batch size, *self.input_shape)
            actions: A tensor of shape (self.n_envs * total buffer batch size, self.n_actions)
            new_states: A tensor of shape (self.n_envs * total buffer batch size, *self.input_shape)
            dones: A tensor of shape (self.n_envs * total buffer batch size)
            rewards: A tensor of shape (self.n_envs * total buffer batch size)

        Returns:
            None
        """
        with tf.GradientTape(True) as tape:
            noise = (
                tf.random.normal(
                    (self.buffers[0].batch_size * self.n_envs, self.n_actions)
                )
                * self.policy_noise_coef
            )
            noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
            new_actions = tf.clip_by_value(
                self.target_actor(new_states) + noise, -1.0, 1.0
            )
            target_critic_input = tf.concat(
                [tf.cast(new_states, tf.float64), tf.cast(new_actions, tf.float64)], 1
            )
            target_value1 = self.target_critic1(target_critic_input)
            target_value2 = self.target_critic2(target_critic_input)
            target_value = tf.minimum(target_value1, target_value2)
            target_value = rewards + tf.stop_gradient(
                (1 - dones) * self.gamma * target_value
            )
            critic_input = tf.concat([states, actions], 1)
            value1 = self.critic1(critic_input)
            value2 = self.critic2(critic_input)
            critic1_loss, critic2_loss = MSE(value1, target_value), MSE(
                value2, target_value
            )
        self.critic1.optimizer.minimize(
            critic1_loss, self.critic1.trainable_variables, tape=tape
        )
        self.critic2.optimizer.minimize(
            critic2_loss, self.critic2.trainable_variables, tape=tape
        )
