import numpy as np
import tensorflow as tf
from xagents import PPO


class TRPO(PPO):
    """
    Trust Region Policy Optimization.
    https://arxiv.org/abs/1502.05477
    """

    def __init__(
        self,
        envs,
        actor_model,
        critic_model,
        max_kl=1e-3,
        cg_iterations=10,
        cg_residual_tolerance=1e-10,
        cg_damping=1e-3,
        actor_iterations=10,
        critic_iterations=3,
        fvp_n_steps=5,
        **kwargs,
    ):
        """
        Initialize TRPO agent.
        Args:
            envs: A list of gym environments.
            actor_model: Actor separate model as a tf.keras.Model.
            critic_model: Critic separate model as a tf.keras.Model.
            max_kl: Maximum KL divergence used for calculating Lagrange multiplier
                and actor weight updates.
            cg_iterations: Gradient conjugation maximum number of iterations per train step.
            cg_residual_tolerance: Gradient conjugation residual tolerance parameter,
                which controls proceeding/stopping iterations.
            cg_damping: Gradient conjugation damping parameter used for calculation
                of Fisher vector product(FVP).
            actor_iterations: Maximum number of actor optimizations steps per train step.
            critic_iterations: Maximum number of critic optimizations steps per train step.
            fvp_n_steps: Value used to skip every n-frames used to calculate FVP
            **kwargs: kwargs passed to super classes.
        """
        super(TRPO, self).__init__(
            envs,
            actor_model,
            **kwargs,
        )
        self.output_models.append(critic_model)
        self.old_actor = tf.keras.models.clone_model(self.model)
        self.actor = self.model
        self.critic = critic_model
        self.cg_iterations = cg_iterations
        self.cg_residual_tolerance = cg_residual_tolerance
        self.cg_damping = cg_damping
        self.max_kl = max_kl
        self.critic_iterations = critic_iterations
        self.actor_iterations = actor_iterations
        self.fvp_n_steps = fvp_n_steps

    @staticmethod
    def flat_to_weights(flat, trainable_variables, in_place=False):
        """
        Convert a flat tensor which should be of the same size as flattened
        `trainable_variables` concatenated to a list of variables and assign
        the new values(optional) to previous `trainable_variables` values.
        Args:
            flat: Flat tensor that is compatible with the given trainable variables
                shapes.
            trainable_variables: A list of tf.Variable objects that should be
                compatible with the given flat tensor.
            in_place: If True, the new values will be assigned to given variables.

        Returns:
            A list of reshaped variables/ an empty list if `in_place`.
        """
        updated_trainable_variables = []
        start_idx = 0
        for trainable_variable in trainable_variables:
            shape = trainable_variable.shape
            flat_size = tf.math.reduce_prod(shape)
            updated_trainable_variable = tf.reshape(
                flat[start_idx : start_idx + flat_size], shape
            )
            if in_place:
                trainable_variable.assign(updated_trainable_variable)
            else:
                updated_trainable_variables.append(updated_trainable_variable)
            start_idx += flat_size
        return updated_trainable_variables

    @staticmethod
    def weights_to_flat(to_flatten, trainable_variables=None):
        """
        Flatten and concatenate a list of gradients/tf.Variable objects.
        Args:
            to_flatten: A list of tf.Variable objects.
            trainable_variables: If specified and the list of gradients
                has None values, these values will be replaced by zeros
                in the flat output.

        Returns:
            Flat tensor.
        """
        if not trainable_variables:
            to_concat = [tf.reshape(non_flat, [-1]) for non_flat in to_flatten]
        else:
            to_concat = [
                tf.reshape(
                    non_flat
                    if non_flat is not None
                    else tf.zeros_like(trainable_variable),
                    [-1],
                )
                for (non_flat, trainable_variable) in zip(
                    to_flatten, trainable_variables
                )
            ]
        return tf.concat(to_concat, 0)

    def calculate_fvp(self, flat_tangent, states):
        """
        Calculate Fisher vector product.
        Args:
            flat_tangent: Flattened gradients obtained by `self.weights_to_flat()`
            states: States tensor expected by the actor model.

        Returns:
            FVP as a flat tensor.
        """
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                kl_divergence, *_ = self.calculate_kl_divergence(states)
            kl_grads = tape1.gradient(kl_divergence, self.actor.trainable_variables)
            tangents = self.flat_to_weights(
                flat_tangent, self.actor.trainable_variables
            )
            gvp = tf.add_n(
                [
                    tf.reduce_sum(grad * tangent)
                    for (grad, tangent) in zip(kl_grads, tangents)
                ]
            )
        hessians_products = tape2.gradient(gvp, self.actor.trainable_variables)
        return (
            self.weights_to_flat(hessians_products, self.actor.trainable_variables)
            + self.cg_damping * flat_tangent
        )

    def conjugate_gradients(self, flat_grads, states):
        """
        Get conjugated gradients.
        Args:
            flat_grads: Flat gradient tensor.
            states: States tensor expected by the actor model.

        Returns:
            Conjugated gradients as a flat tensor.
        """
        p = tf.identity(flat_grads)
        r = tf.identity(flat_grads)
        x = tf.zeros_like(flat_grads)
        r_dot_r = tf.tensordot(r, r, 1)
        iterations = 0
        while tf.less(iterations, self.cg_iterations) and tf.greater(
            r_dot_r, self.cg_residual_tolerance
        ):
            z = self.calculate_fvp(p, states)
            v = r_dot_r / tf.tensordot(p, z, 1)
            x += v * p
            r -= v * z
            new_r_dot_r = tf.tensordot(r, r, 1)
            mu = new_r_dot_r / r_dot_r
            p = r + mu * p
            r_dot_r = new_r_dot_r
            iterations += 1
        return x

    def calculate_kl_divergence(self, states):
        """
        Calculate probability distribution of both new and old actor models
        and calculate Kullback–Leibler divergence.
        Args:
            states: States tensor expected by the actor models.

        Returns:
            Mean KL divergence, old distribution and new distribution.
        """
        old_actor_output = self.get_model_outputs(
            states, [self.old_actor, self.critic]
        )[4]
        new_actor_output = self.get_model_outputs(states, self.output_models)[4]
        old_distribution = self.get_distribution(old_actor_output)
        new_distribution = self.get_distribution(new_actor_output)
        return (
            tf.reduce_mean(old_distribution.kl_divergence(new_distribution)),
            old_distribution,
            new_distribution,
        )

    def calculate_losses(self, states, actions, advantages):
        """
        Calculate surrogate loss and KL divergence.
        Args:
            states: States tensor expected by the actor models.
            actions: Tensor of actions taken.
            advantages: Tensor of calculated advantages.

        Returns:
            Surrogate loss and KL divergence.
        """
        (
            kl_divergence,
            old_distribution,
            new_distribution,
        ) = self.calculate_kl_divergence(states)
        entropy = tf.reduce_mean(new_distribution.entropy())
        entropy_loss = self.entropy_coef * entropy
        ratio = tf.exp(
            new_distribution.log_prob(actions) - old_distribution.log_prob(actions)
        )
        surrogate_gain = tf.reduce_mean(ratio * advantages)
        surrogate_loss = surrogate_gain + entropy_loss
        return surrogate_loss, kl_divergence

    def at_step_start(self):
        """
        Execute steps that will run before self.train_step() which updates
        old actor weights with the optimized weights.

        Returns:
            None
        """
        self.old_actor.set_weights(self.actor.get_weights())

    def update_actor_weights(
        self,
        flat_weights,
        full_step,
        surrogate_loss,
        states,
        actions,
        advantages,
    ):
        """
        Calculate losses, and run actor iterations until reaching `actor_iterations`
        specified earlier or reaching stop/ok conditions.
        Args:
            flat_weights: A tensor of flattened actor weights.
            full_step: FVP / Lagrange multiplier.
            surrogate_loss: The lower bound of the original objective - the expected
                cumulative return of the policy
            states: States tensor expected by the actor models.
            actions: Tensor of actions taken.
            advantages: Tensor of calculated advantages.

        Returns:
            None
        """
        learning_rate = 1.0
        for _ in range(self.actor_iterations):
            updated_weights = flat_weights + full_step * learning_rate
            self.flat_to_weights(updated_weights, self.actor.trainable_variables, True)
            losses = new_surrogate_loss, new_kl_divergence = self.calculate_losses(
                states, actions, advantages
            )
            improvement = new_surrogate_loss - surrogate_loss
            ok_conditions = [
                np.isfinite(losses).all(),
                new_kl_divergence <= self.max_kl * 1.5,
                improvement > 0,
            ]
            if all(ok_conditions):
                break
            learning_rate *= 0.5
        else:
            self.flat_to_weights(flat_weights, self.actor.trainable_variables, True)

    def update_critic_weights(self, states, returns):
        """
        Calculate value loss and apply gradients.
        Args:
            states: States tensor expected by the critic model.
            returns: The cumulative returns tensor.

        Returns:
            None
        """
        for _ in range(self.critic_iterations):
            for (states_mb, returns_mb) in self.get_mini_batches(states, returns):
                with tf.GradientTape() as tape:
                    values = self.get_model_outputs(states_mb, self.output_models)[2]
                    value_loss = tf.reduce_mean(tf.square(values - returns_mb))
                grads = tape.gradient(value_loss, self.critic.trainable_variables)
                self.critic.optimizer.apply_gradients(
                    zip(grads, self.critic.trainable_variables)
                )

    @tf.function
    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.envs, batching and gradient updates.

        Returns:
            None
        """
        states, actions, returns, values, _ = tf.numpy_function(
            self.get_batch, [], 5 * [tf.float32]
        )
        advantages = returns - values
        advantages = (advantages - tf.reduce_mean(advantages)) / tf.math.reduce_std(
            advantages
        )
        with tf.GradientTape() as tape:
            surrogate_loss, kl_divergence = self.calculate_losses(
                states, actions, advantages
            )
        flat_grads = self.weights_to_flat(
            tape.gradient(surrogate_loss, self.actor.trainable_variables),
            self.actor.trainable_variables,
        )
        step_direction = self.conjugate_gradients(
            flat_grads, states[:: self.fvp_n_steps]
        )
        shs = 0.5 * tf.tensordot(
            step_direction,
            self.calculate_fvp(step_direction, states[:: self.fvp_n_steps]),
            1,
        )
        lagrange_multiplier = tf.math.sqrt(shs / self.max_kl)
        full_step = step_direction / lagrange_multiplier
        pre_actor_weights = self.weights_to_flat(
            self.actor.trainable_variables,
        )
        tf.numpy_function(
            self.update_actor_weights,
            [
                pre_actor_weights,
                full_step,
                surrogate_loss,
                states,
                actions,
                advantages,
            ],
            [],
        )
        self.update_critic_weights(states, returns)
