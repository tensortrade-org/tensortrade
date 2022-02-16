import os
import random
from abc import ABC
from collections import deque
from datetime import timedelta
from pathlib import Path
from time import perf_counter, sleep

import cv2
import gym
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
import wandb
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from termcolor import colored
from xagents.utils.common import write_from_dict


class BaseAgent(ABC):
    """
    Base class for various types of agents.
    """

    def __init__(
        self,
        envs,
        model,
        checkpoints=None,
        reward_buffer_size=100,
        n_steps=1,
        gamma=0.99,
        display_precision=2,
        seed=None,
        log_frequency=None,
        history_checkpoint=None,
        plateau_reduce_factor=0.9,
        plateau_reduce_patience=10,
        early_stop_patience=3,
        divergence_monitoring_steps=None,
        quiet=False,
        trial=None,
    ):
        """
        Initialize base settings.
        Args:
            envs: A list of gym environments.
            model: tf.keras.models.Model that is expected to be compiled
                with an optimizer before training starts.
            checkpoints: A list of paths to .tf filenames under which the trained model(s)
                will be saved.
            reward_buffer_size: Size of the reward buffer that will hold the last n total
                rewards which will be used for calculating the mean reward.
            n_steps: n-step transition for example given s1, s2, s3, s4 and n_step = 4,
                transition will be s1 -> s4 (defaults to 1, s1 -> s2)
            gamma: Discount factor used for gradient updates.
            display_precision: Decimal precision for display purposes.
            seed: Random seed passed to random.seed(), np.random.seed(), tf.random.seed(),
                env.seed()
            log_frequency: Interval of done games to display progress after each,
                defaults to the number of environments given if not specified.
            history_checkpoint: Path to .parquet file to which episode history will be saved.
            plateau_reduce_patience: int, Maximum times of non-improving consecutive model checkpoints.
            plateau_reduce_factor: Factor by which the learning rates of all models in
                self.output_models are multiplied when plateau_reduce_patience is consecutively
                reached / exceeded.
            early_stop_patience: Number of times plateau_reduce_patience is consecutively
                reached / exceeded.
            divergence_monitoring_steps: Number of steps at which reduce on plateau,
                and early stopping start monitoring.
            quiet: If True, all agent messages will be silenced.
            trial: optuna.trial.Trial
        """
        assert envs, 'No environments given'
        self.n_envs = len(envs)
        self.envs = envs
        self.model = model
        self.checkpoints = checkpoints
        self.total_rewards = deque(maxlen=reward_buffer_size)
        self.n_steps = n_steps
        self.gamma = gamma
        self.display_precision = display_precision
        self.seed = seed
        self.output_models = [self.model]
        self.log_frequency = log_frequency or self.n_envs
        self.id = self.__module__.split('.')[1]
        self.history_checkpoint = history_checkpoint
        self.plateau_reduce_factor = plateau_reduce_factor
        self.plateau_reduce_patience = plateau_reduce_patience
        self.early_stop_patience = early_stop_patience
        self.divergence_monitoring_steps = divergence_monitoring_steps
        self.quiet = quiet
        self.trial = trial
        self.reported_rewards = 0
        self.plateau_count = 0
        self.early_stop_count = 0
        self.target_reward = None
        self.max_steps = None
        self.input_shape = self.envs[0].observation_space.shape
        self.n_actions = None
        self.best_reward = -float('inf')
        self.mean_reward = -float('inf')
        self.states = [np.array(0)] * self.n_envs
        self.dones = [False] * self.n_envs
        self.steps = 0
        self.frame_speed = 0
        self.last_reset_step = 0
        self.training_start_time = None
        self.last_reset_time = None
        self.games = 0
        self.episode_rewards = np.zeros(self.n_envs)
        self.done_envs = 0
        self.supported_action_spaces = Box, Discrete
        if seed:
            self.set_seeds(seed)
        self.reset_envs()
        self.set_action_count()
        self.img_inputs = len(self.states[0].shape) >= 2
        self.display_titles = (
            'time',
            'steps',
            'games',
            'speed',
            'mean reward',
            'best reward',
        )

    def assert_valid_env(self, env, valid_type):
        """
        Assert the right type of environment is passed to an agent.
        Args:
            env: gym environment.
            valid_type: gym.spaces class.

        Returns:
            None
        """
        assert isinstance(env.action_space, valid_type), (
            f'Invalid environment: {env.spec.id}. {self.__class__.__name__} supports '
            f'environments with a {valid_type} action space only, got {env.action_space}'
        )

    def display_message(self, *args, **kwargs):
        """
        Display messages to the console.
        Args:
            *args: args passed to print()
            **kwargs: kwargs passed to print()

        Returns:
            None
        """
        if not self.quiet:
            print(*args, **kwargs)

    def set_seeds(self, seed):
        """
        Set random seeds for numpy, tensorflow, random, gym
        Args:
            seed: int, random seed.

        Returns:
            None
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)
        for env in self.envs:
            env.seed(seed)
            env.action_space.seed(seed)
        os.environ['PYTHONHASHSEED'] = f'{seed}'
        random.seed(seed)

    def reset_envs(self):
        """
        Reset all environments in self.envs and update self.states

        Returns:
            None
        """
        for i, env in enumerate(self.envs):
            self.states[i] = env.reset()

    def set_action_count(self):
        """
        Set `self.n_actions` to the number of actions for discrete
        environments or to the shape of the action space for continuous.
        """
        action_space = self.envs[0].action_space
        assert (
            type(action_space) in self.supported_action_spaces
        ), f'Expected one of {self.supported_action_spaces}, got {action_space}'
        if isinstance(action_space, Discrete):
            self.n_actions = action_space.n
        if isinstance(action_space, Box):
            self.n_actions = action_space.shape[0]

    def check_checkpoints(self):
        """
        Ensure the number of given checkpoints matches the number of output models.

        Returns:
            None
        """
        n_models = len(self.output_models)
        n_checkpoints = len(self.checkpoints)
        assert n_models == n_checkpoints, (
            f'Expected {n_models} checkpoints for {n_models} '
            f'given output models, got {n_checkpoints}'
        )

    def checkpoint(self):
        """
        Save model weights if current reward > best reward.

        Returns:
            None
        """
        if self.mean_reward > self.best_reward:
            self.plateau_count = 0
            self.early_stop_count = 0
            self.display_message(
                f'Best reward updated: {colored(self.best_reward, "red")} -> '
                f'{colored(self.mean_reward, "green")}'
            )
            if self.checkpoints:
                for model, checkpoint in zip(self.output_models, self.checkpoints):
                    model.save_weights(checkpoint)
        self.best_reward = max(self.mean_reward, self.best_reward)

    def display_metrics(self):
        """
        Display progress metrics to the console when environments complete a full episode each.
        Metrics consist of:
            - time: Time since training started.
            - steps: Time steps so far.
            - games: Finished games / episodes that resulted in a terminal state.
            - speed: Frame speed/s
            - mean reward: Mean game total rewards.
            - best reward: Highest total episode score obtained.

        Returns:
            None
        """
        display_values = (
            timedelta(seconds=perf_counter() - self.training_start_time),
            self.steps,
            self.games,
            f'{round(self.frame_speed)} steps/s',
            self.mean_reward,
            self.best_reward,
        )
        display = (
            f'{title}: {value}'
            for title, value in zip(self.display_titles, display_values)
        )
        self.display_message(', '.join(display))

    def update_metrics(self):
        """
        Update progress metrics which consist of last reset step and time used
        for calculation of fps, and update mean and best rewards. The model is
        saved if there is a checkpoint path specified.

        Returns:
            None
        """
        self.checkpoint()
        if (
            self.divergence_monitoring_steps
            and self.steps >= self.divergence_monitoring_steps
            and self.mean_reward <= self.best_reward
        ):
            self.plateau_count += 1
        if self.plateau_count >= self.plateau_reduce_patience:
            current_lr, new_lr = None, None
            for model in self.output_models:
                current_lr = model.optimizer.learning_rate
                new_lr = current_lr * self.plateau_reduce_factor
            self.display_message(
                f'Learning rate reduced {current_lr.numpy()} ' f'-> {new_lr.numpy()}'
            )
            current_lr.assign(new_lr)
            self.plateau_count = 0
            self.early_stop_count += 1
        self.frame_speed = (self.steps - self.last_reset_step) / (
            perf_counter() - self.last_reset_time
        )
        self.last_reset_step = self.steps
        self.mean_reward = np.around(
            np.mean(self.total_rewards), self.display_precision
        )

    def report_rewards(self):
        """
        Report intermediate rewards or raise an exception to
        prune current trial.

        Returns:
            None

        Raises:
            optuna.exceptions.TrialPruned
        """
        self.trial.report(np.mean(self.total_rewards), self.reported_rewards)
        self.reported_rewards += 1
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    def check_episodes(self):
        """
        Check environment done counts to display progress and update metrics.

        Returns:
            None
        """
        if self.done_envs >= self.log_frequency:
            self.update_metrics()
            if self.trial:
                self.report_rewards()
            self.last_reset_time = perf_counter()
            self.display_metrics()
            self.done_envs = 0

    def training_done(self):
        """
        Check whether a target reward or maximum number of steps is reached.

        Returns:
            bool
        """
        if self.early_stop_count >= self.early_stop_patience:
            self.display_message(f'Early stopping')
            return True
        if self.target_reward and self.mean_reward >= self.target_reward:
            self.display_message(f'Reward achieved in {self.steps} steps')
            return True
        if self.max_steps and self.steps >= self.max_steps:
            self.display_message(f'Maximum steps exceeded')
            return True
        return False

    def concat_buffer_samples(self):
        """
        Concatenate samples drawn from each environment respective buffer.
        Args:

        Returns:
            A list of concatenated samples.
        """
        if hasattr(self, 'buffers'):
            batches = []
            for i, env in enumerate(self.envs):
                buffer = self.buffers[i]
                batch = buffer.get_sample()
                batches.append(batch)
            dtypes = (
                self.batch_dtypes
                if hasattr(self, 'batch_dtypes')
                else [np.float32 for _ in range(len(batches[0]))]
            )
            if len(batches) > 1:
                return [
                    np.concatenate(item).astype(dtype)
                    for (item, dtype) in zip(zip(*batches), dtypes)
                ]
            return [item.astype(dtype) for (item, dtype) in zip(batches[0], dtypes)]

    def update_history(self, episode_reward):
        """
        Write 1 episode stats to .parquet history checkpoint.
        Args:
            episode_reward: int, a finished episode reward

        Returns:
            None
        """
        data = {
            'mean_reward': [self.mean_reward],
            'best_reward': [self.best_reward],
            'episode_reward': [episode_reward],
            'step': [self.steps],
            'time': [perf_counter() - self.training_start_time],
        }
        write_from_dict(data, self.history_checkpoint)

    def step_envs(self, actions, get_observation=False, store_in_buffers=False):
        """
        Step environments in self.envs, update metrics (if any done games)
            and return / store results.
        Args:
            actions: An iterable of actions to execute by environments.
            get_observation: If True, a list of [states, actions, rewards, dones, new_states]
                of length self.n_envs each will be returned.
            store_in_buffers: If True, each observation is saved separately in respective buffer.

        Returns:
            A list of observations as numpy arrays or an empty list.
        """
        observations = []
        for (
            (i, env),
            action,
            *items,
        ) in zip(enumerate(self.envs), actions):
            state = self.states[i]
            new_state, reward, done, _ = env.step(action)
            self.states[i] = new_state
            self.dones[i] = done
            self.episode_rewards[i] += reward
            observation = state, action, reward, done, new_state
            if store_in_buffers and hasattr(self, 'buffers'):
                self.buffers[i].append(*observation)
            if get_observation:
                observations.append(observation)
            if done:
                if self.history_checkpoint:
                    self.update_history(self.episode_rewards[i])
                self.done_envs += 1
                self.total_rewards.append(self.episode_rewards[i])
                self.games += 1
                self.episode_rewards[i] = 0
                self.states[i] = env.reset()
            self.steps += 1
        return [np.array(item, np.float32) for item in zip(*observations)]

    def init_from_checkpoint(self):
        """
        Load previous training session metadata and update agent metrics
            to go from there.

        Returns:
            None
        """
        previous_history = pd.read_parquet(self.history_checkpoint)
        expected_columns = {
            'time',
            'mean_reward',
            'best_reward',
            'step',
            'episode_reward',
        }
        assert (
            set(previous_history.columns) == expected_columns
        ), f'Expected the following columns: {expected_columns}, got {set(previous_history.columns)}'
        last_row = previous_history.loc[previous_history['time'].idxmax()]
        self.mean_reward = last_row['mean_reward']
        self.best_reward = previous_history['best_reward'].max()
        history_start_steps = last_row['step']
        history_start_time = last_row['time']
        self.training_start_time = perf_counter() - history_start_time
        self.last_reset_step = self.steps = int(history_start_steps)
        self.total_rewards.append(last_row['episode_reward'])
        self.games = previous_history.shape[0]

    def init_training(self, target_reward, max_steps, monitor_session):
        """
        Initialize training start time, wandb session & models (self.model / self.target_model)
        Args:
            target_reward: Total reward per game value that whenever achieved,
                the training will stop.
            max_steps: Maximum time steps, if exceeded, the training will stop.
            monitor_session: Wandb session name.

        Returns:
            None
        """
        self.target_reward = target_reward
        self.max_steps = max_steps
        if monitor_session:
            wandb.init(name=monitor_session)
        if self.checkpoints:
            self.check_checkpoints()
        self.training_start_time = perf_counter()
        self.last_reset_time = perf_counter()
        if self.history_checkpoint and Path(self.history_checkpoint).exists():
            self.init_from_checkpoint()

    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.envs, batching and gradient updates.

        Returns:
            None
        """
        raise NotImplementedError(
            f'train_step() should be implemented by {self.__class__.__name__} subclasses'
        )

    def get_model_outputs(self, inputs, models, training=True):
        """
        Get single or multiple model outputs.
        Args:
            inputs: Inputs as tensors / numpy arrays that are expected
                by the given model(s).
            models: A tf.keras.Model or a list of tf.keras.Model(s)
            training: `training` parameter passed to model call.

        Returns:
            Outputs as a list in case of multiple models or any other shape
            that is expected from the given model(s).
        """
        if self.img_inputs:
            inputs = tf.cast(inputs, tf.float32) / 255.0
        if isinstance(models, tf.keras.models.Model):
            return models(inputs, training=training)
        elif len(models) == 1:
            return models[0](inputs, training=training)
        return [sub_model(inputs, training=training) for sub_model in models]

    def at_step_start(self):
        """
        Execute steps that will run before self.train_step().

        Returns:
            None
        """
        pass

    def at_step_end(self):
        """
        Execute steps that will run after self.train_step().

        Returns:
            None
        """
        pass

    def get_states(self):
        """
        Get most recent states.

        Returns:
            self.states as numpy array.
        """
        return np.array(self.states)

    def get_dones(self):
        """
        Get most recent game statuses.

        Returns:
            self.dones as numpy array.
        """
        return np.array(self.dones, np.float32)

    @staticmethod
    def concat_step_batches(*args):
        """
        Concatenate n-step batches.
        Args:
            *args: A list of numpy arrays which will be concatenated separately.

        Returns:
            A list of concatenated numpy arrays.
        """
        concatenated = []
        for arg in args:
            if len(arg.shape) == 1:
                arg = np.expand_dims(arg, -1)
            concatenated.append(arg.swapaxes(0, 1).reshape(-1, *arg.shape[2:]))
        return concatenated

    def fit(
        self,
        target_reward=None,
        max_steps=None,
        monitor_session=None,
    ):
        """
        Common training loop shared by subclasses, monitors training status
        and progress, performs all training steps, updates metrics, and logs progress.
        Args:
            target_reward: Target reward, if achieved, the training will stop
            max_steps: Maximum number of steps, if reached the training will stop.
            monitor_session: Session name to use for monitoring the training with wandb.

        Returns:
            None
        """
        assert (
            target_reward or max_steps
        ), '`target_reward` or `max_steps` should be specified when fit() is called'
        self.init_training(target_reward, max_steps, monitor_session)
        while True:
            self.check_episodes()
            if self.training_done():
                break
            self.at_step_start()
            self.train_step()
            self.at_step_end()

    def play(
        self,
        video_dir=None,
        render=False,
        frame_dir=None,
        frame_delay=0.0,
        max_steps=None,
        action_idx=0,
        frame_frequency=1,
    ):
        """
        Play and display a game.
        Args:
            video_dir: Path to directory to save the resulting game video.
            render: If True, the game will be displayed.
            frame_dir: Path to directory to save game frames.
            frame_delay: Delay between rendered frames.
            max_steps: Maximum environment steps.
            action_idx: Index of action output by self.model
            frame_frequency: If frame_dir is specified, save frames every n frames.

        Returns:
            None
        """
        self.reset_envs()
        env_idx = 0
        total_reward = 0
        env_in_use = self.envs[env_idx]
        if video_dir:
            env_in_use = gym.wrappers.Monitor(env_in_use, video_dir)
            env_in_use.reset()
        steps = 0
        agent_id = self.__module__.split('.')[1]
        for dir_name in (video_dir, frame_dir):
            os.makedirs(dir_name or '.', exist_ok=True)
        while True:
            if max_steps and steps >= max_steps:
                self.display_message(f'Maximum steps {max_steps} exceeded')
                break
            if render:
                env_in_use.render()
                sleep(frame_delay)
            if frame_dir and steps % frame_frequency == 0:
                frame = cv2.cvtColor(
                    env_in_use.render(mode='rgb_array'), cv2.COLOR_BGR2RGB
                )
                cv2.imwrite(os.path.join(frame_dir, f'{steps:05d}.jpg'), frame)
            if hasattr(self, 'actor') and agent_id in ['td3', 'ddpg']:
                action = self.actor(self.get_states())[env_idx]
            else:
                action = self.get_model_outputs(
                    self.get_states(), self.output_models, False
                )[action_idx][env_idx].numpy()
            self.states[env_idx], reward, done, _ = env_in_use.step(action)
            total_reward += reward
            if done:
                self.display_message(f'Total reward: {total_reward}')
                break
            steps += 1


class OnPolicy(BaseAgent, ABC):
    """
    Base class for on-policy agents.
    """

    def __init__(self, envs, model, **kwargs):
        """
        Initialize on-policy agent.
        Args:
            envs: A list of gym environments.
            model: tf.keras.models.Model that is expected to be compiled
                with an optimizer before training starts.
            **kwargs: kwargs passed to BaseAgent.
        """
        super(OnPolicy, self).__init__(envs, model, **kwargs)


class OffPolicy(BaseAgent, ABC):
    """
    Base class for off-policy agents.
    """

    def __init__(
        self,
        envs,
        model,
        buffers,
        **kwargs,
    ):
        """
        Initialize off-policy agent.
        Args:
            envs: A list of gym environments.
            model: tf.keras.models.Model that is expected to be compiled
                with an optimizer before training starts.
            buffers: A list of replay buffer objects whose length should match
                `envs`s'.
            **kwargs: kwargs passed to BaseAgent.
        """
        super(OffPolicy, self).__init__(envs, model, **kwargs)
        assert len(envs) == len(buffers), (
            f'Expected equal env and replay buffer sizes, got {self.n_envs} '
            f'and {len(buffers)}'
        )
        self.buffers = buffers

    def fill_buffers(self):
        """
        Fill each buffer in self.buffers up to its initial size.

        Returns:
            None
        """
        total_size = sum(buffer.initial_size for buffer in self.buffers)
        sizes = {}
        for i, env in enumerate(self.envs):
            buffer = self.buffers[i]
            state = self.states[i]
            while buffer.current_size < buffer.initial_size:
                action = env.action_space.sample()
                new_state, reward, done, _ = env.step(action)
                buffer.append(state, action, reward, done, new_state)
                state = new_state
                if done:
                    state = env.reset()
                sizes[i] = buffer.current_size
                filled = sum(sizes.values())
                complete = round((filled / total_size) * 100, self.display_precision)
                self.display_message(
                    f'\rFilling replay buffer {i + 1}/{self.n_envs} ==> {complete}% | '
                    f'{filled}/{total_size}',
                    end='',
                )
        self.display_message('')
        self.reset_envs()

    def fit(
        self,
        target_reward=None,
        max_steps=None,
        monitor_session=None,
    ):
        """
        Common training loop shared by subclasses, monitors training status
        and progress, performs all training steps, updates metrics, and logs progress.
        ** Additionally, replay buffers are pre-filled before training starts **
        Args:
            target_reward: Target reward, if achieved, the training will stop
            max_steps: Maximum number of steps, if reached the training will stop.
            monitor_session: Session name to use for monitoring the training with wandb.

        Returns:
            None
        """
        self.fill_buffers()
        super(OffPolicy, self).fit(target_reward, max_steps, monitor_session)
