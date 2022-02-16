import random
from time import perf_counter

import gym
import numpy as np
import optuna
import pandas as pd
import pytest
import tensorflow as tf
import wandb
import xagents
from gym.spaces import Discrete
from optuna import Trial
from optuna.distributions import UniformDistribution
from xagents import A2C, ACER, DDPG, DQN, PPO, TD3, TRPO
from xagents.base import BaseAgent, OffPolicy, OnPolicy
from xagents.utils.buffers import ReplayBuffer1
from xagents.utils.common import (create_buffers, create_models, get_wandb_key,
                                  write_from_dict)


class DummyAgent(BaseAgent):
    """
    Lightweight agent for testing time consuming tasks.
    """

    def __init__(self, envs, model, **kwargs):
        """
        Initialize dummy agent.
        Args:
            envs: A list of gym environments which are meant as placeholders.
            model: tf.keras.Model which serves as a placeholder.
            **kwargs: kwargs to be passed to super classes.
        """
        super(DummyAgent, self).__init__(envs, model, **kwargs)

    def at_step_start(self):
        """
        Execute steps that will run before self.train_step()
        which consist of printing a few words for testing purposes.

        Returns:
            None
        """
        print(f'step starting')

    def train_step(self):
        """
        A lightweight train step for test convenience purposes.

        Returns:
            None
        """
        self.steps += 1
        print(f'train_step')

    def at_step_end(self):
        """
        Execute steps that will run after self.train_step()
        which consist of printing a few words for testing purposes.

        Returns:
            None
        """
        print(f'step ending')


@pytest.mark.usefixtures('envs', 'envs2', 'model', 'buffers', 'study')
class TestBase:
    """
    Tests for base agents and their methods.
    """

    model_counts = {
        DDPG: 2,
        TD3: 3,
        TRPO: 2,
        A2C: 1,
        ACER: 1,
        DQN: 1,
        PPO: 1,
        BaseAgent: 1,
        OnPolicy: 1,
        OffPolicy: 1,
    }

    def get_agent_kwargs(
        self, agent, envs=None, model=None, buffers=None, critic_model=None
    ):
        """
        Construct agent required kwargs according to the given agent.
        Args:
            agent: OnPolicy/OffPolicy subclass.
            envs: A list of gym environments, if not specified, the default
                class envs will be returned.
            model: tf.keras.Model, if not specified, the default class model
                will be returned.
            buffers: A list of replay buffers, if not specified, the default
                class buffers will be returned.
            critic_model: tf.keras.Model, if not specified, the default class
                model will be returned.

        Returns:
            agent_kwargs.
        """
        valid_envs = self.envs if agent not in [TD3, DDPG] else self.envs2
        agent_kwargs = {'envs': envs if envs is not None else valid_envs}
        buffers = buffers or self.buffers
        if self.model_counts[agent] > 1:
            agent_kwargs['actor_model'] = model or tf.keras.models.clone_model(
                self.model
            )
            agent_kwargs['critic_model'] = critic_model or tf.keras.models.clone_model(
                self.model
            )
            agent_kwargs['actor_model'].compile('adam')
            agent_kwargs['critic_model'].compile('adam')
        else:
            agent_kwargs['model'] = model or self.model
        if agent == xagents.ACER:
            for buffer in buffers:
                buffer.batch_size = 1
            agent_kwargs['buffers'] = buffers
        if issubclass(agent, OffPolicy) or agent == OffPolicy:
            agent_kwargs['buffers'] = buffers
        return agent_kwargs

    def test_no_envs(self, agent):
        """
        Test no given environments.
        Args:
            agent: OnPolicy/OffPolicy subclass.
        """
        with pytest.raises(AssertionError, match=r'No environments given'):
            agent_kwargs = self.get_agent_kwargs(agent, [])
            agent(**agent_kwargs)

    def test_display_message(self, agent, capsys):
        agent = agent(**self.get_agent_kwargs(agent))
        message = 'displayed message'
        agent.display_message(message)
        assert message in capsys.readouterr().out
        agent.quiet = True
        agent.display_message(message)
        assert not capsys.readouterr().out

    def test_seeds(self, agent):
        """
        Test random seeds (random/numpy/tensorflow/gym)
        Args:
            agent: OnPolicy/OffPolicy subclass.
        """
        test_seed = 1
        agent = agent(**self.get_agent_kwargs(agent))
        results1, results2 = set(), set()
        test_range = 1000000
        for results in [results1, results2]:
            agent.set_seeds(test_seed)
            results.add(random.randint(0, test_range))
            results.add(int(np.random.randint(0, test_range, 1)))
            results.add(int(tf.random.uniform([1], maxval=test_range)))
            sample = agent.envs[0].action_space.sample()
            if isinstance(sample, np.ndarray):
                sample = tuple(sample)
            results.add(sample)
        assert results1 == results2

    @pytest.mark.parametrize(
        'env_ids',
        [
            ('RiverraidNoFrameskip-v4', 'MountainCarContinuous-v0'),
            ('Robotank-ramDeterministic-v0', 'Pendulum-v1'),
            ('BeamRiderDeterministic-v4', 'LunarLanderContinuous-v2'),
            ('PongNoFrameskip-v4', 'BipedalWalker-v3'),
            ('GravitarNoFrameskip-v0', 'BipedalWalkerHardcore-v3'),
            ('Freeway-ramDeterministic-v0', 'CarRacing-v0'),
        ],
    )
    def test_n_actions(self, env_ids, agent):
        """
        Test if agent initializes the correct n_actions attribute.
        Args:
            env_ids: discrete/box environment id pair.
            agent: OnPolicy/OffPolicy subclass.
        """
        env_id = env_ids[0] if agent not in [TD3, DDPG] else env_ids[1]
        env = gym.make(env_id)
        agent = agent(
            **self.get_agent_kwargs(agent, [env for _ in range(len(self.envs))])
        )
        if isinstance(env.action_space, Discrete):
            assert agent.n_actions == env.action_space.n
        else:
            assert agent.n_actions == env.action_space.shape[0]

    def test_init_from_checkpoint(self, agent, tmp_path):
        """
        Test initialization from history checkpoint, and ensure
        metrics are retrieved with the proper values.
        Args:
            agent: OnPolicy/OffPolicy subclass.
            tmp_path: pathlib.PosixPath
        """
        columns = [
            'time',
            'mean_reward',
            'best_reward',
            'step',
            'episode_reward',
        ]
        invalid_data = [{'a': [1], 'b': [2]}, {'a': [3], 'b': [4]}]
        invalid_checkpoint = tmp_path / 'invalid_checkpoint.parquet'
        for invalid_item in invalid_data:
            write_from_dict(invalid_item, invalid_checkpoint)
        agent = agent(
            **self.get_agent_kwargs(agent), history_checkpoint=invalid_checkpoint
        )
        with pytest.raises(AssertionError, match=r'Expected the following columns'):
            agent.init_from_checkpoint()
        history_size = 10
        valid_history = []
        agent.history_checkpoint = valid_checkpoint = (
            tmp_path / 'valid_checkpoint.parquet'
        )
        for size in range(history_size):
            valid_data = {column: [random.randint(0, 100)] for column in columns}
            write_from_dict(valid_data, valid_checkpoint)
            valid_history.append(
                {column: value[0] for (column, value) in valid_data.items()}
            )
        valid_history = pd.DataFrame(valid_history)
        agent.init_from_checkpoint()
        last_row = valid_history.loc[valid_history['time'].idxmax()]
        assert agent.mean_reward == last_row['mean_reward']
        assert agent.best_reward == valid_history['best_reward'].max()
        history_start_steps = last_row['step']
        history_start_time = last_row['time']
        times = agent.training_start_time, perf_counter() - history_start_time
        assert np.isclose(*times, rtol=0.1)
        assert agent.last_reset_step == agent.steps == history_start_steps
        assert np.mean(agent.total_rewards) == last_row['episode_reward']
        assert agent.games == valid_history.shape[0]

    def test_update_history(self, agent, tmp_path):
        """
        Test training history update that should happen after each
        finished episode. Several test episode values are checked in
        the respective file and validated against what's expected.
        Args:
            agent: OnPolicy/OffPolicy subclass.
            tmp_path: pathlib.PosixPath
        """
        history_checkpoint = tmp_path / 'test_checkpoint.parquet'
        agent = agent(
            **self.get_agent_kwargs(agent),
            history_checkpoint=history_checkpoint.as_posix(),
        )
        agent.training_start_time = perf_counter()
        mean_rewards = []
        best_rewards = []
        episode_rewards = []
        steps = []
        times = []
        for _ in range(5):
            mean_rewards.append(random.randint(1, 1000))
            best_rewards.append(random.randint(1, 1000))
            episode_rewards.append(random.randint(1, 1000))
            steps.append(random.randint(1, 1000))
            times.append(perf_counter() - agent.training_start_time)
            agent.mean_reward = mean_rewards[-1]
            agent.best_reward = best_rewards[-1]
            agent.steps = steps[-1]
            agent.update_history(episode_rewards[-1])
        assert history_checkpoint.exists()
        actual = (
            pd.read_parquet(history_checkpoint.as_posix())
            .sort_values('time')
            .drop('time', axis=1)
            .reset_index(drop=True)
        )
        expected = pd.DataFrame([mean_rewards, best_rewards, episode_rewards, steps]).T
        expected.columns = actual.columns
        assert (actual == expected).all().all()

    def test_checkpoint(self, agent, tmp_path, capsys):
        """
        Test display after keywords that are expected after a checkpoint.
        Also test for the presence of the expected checkpoint .tf files
        that are usually saved during training and ensure the number of models
        per agent in self.model_counts matches the one required by the agent,
        otherwise, an error will be raised by the agent.
        Args:
            agent: OnPolicy/OffPolicy subclass.
            tmp_path: pathlib.PosixPath
            capsys: _pytest.capture.CaptureFixture
        """
        checkpoints = {
            (tmp_path / f'test_weights{i}.tf').as_posix()
            for i in range(self.model_counts[agent])
        }
        expected_filenames = set()
        for checkpoint in checkpoints:
            expected_filenames.add(f'{checkpoint}.index')
            expected_filenames.add(f'{checkpoint}.data-00000-of-00001')
        expected_filenames.add((tmp_path / 'checkpoint').as_posix())
        agent = agent(**self.get_agent_kwargs(agent), checkpoints=checkpoints)
        agent.checkpoint()
        assert not capsys.readouterr().out
        agent.mean_reward = 100
        agent.checkpoint()
        assert 'Best reward updated' in capsys.readouterr().out
        assert agent.best_reward == 100
        resulting_files = {item.as_posix() for item in tmp_path.iterdir()}
        assert expected_filenames == resulting_files

    def test_wrong_checkpoints(self, agent):
        """
        Ensure an error is raised for the wrong number of checkpoints.
        Args:
            agent: OnPolicy/OffPolicy subclass.
        """
        agent = agent(
            **self.get_agent_kwargs(agent),
            checkpoints=(self.model_counts[agent] + 1) * ['wrong_ckpt.tf'],
        )
        with pytest.raises(AssertionError, match=r'given output models, got'):
            agent.fit(18)

    @staticmethod
    def assert_progress_displayed(displayed):
        """
        Assert text displayed to the console has the expected keywords.
        Args:
            displayed: str
        """
        for item in (
            'time',
            'steps',
            'games',
            'speed',
            'mean reward',
            'best reward',
        ):
            assert item in displayed

    def test_display_metrics(self, capsys, agent):
        """
        Check metric names are printed to the console.
        Args:
            capsys: _pytest.capture.CaptureFixture
            agent: OnPolicy/OffPolicy subclass.
        """
        agent = agent(**self.get_agent_kwargs(agent))
        agent.training_start_time = perf_counter()
        agent.frame_speed = agent.mean_reward = agent.best_reward = 0
        agent.display_metrics()
        displayed = capsys.readouterr().out
        self.assert_progress_displayed(displayed)

    @pytest.mark.parametrize(
        'steps, total_rewards, mean_reward, best_reward, plateau_count, '
        'expected_fps, expected_mean, learning_rate,'
        'interval, early_stop_count',
        [
            [1000, [100, 130, 150], 100, 99, 0, 500, 126.67, 0.01, 2, 0],
            [2000, [10, 15, 20], 50, 52, 3, 1000, 15, 0.01, 2, 1],
            [600000, [400, 300, 20], 100, 105, 1, 600, 240, 0.05, 1000, 1],
            [100000, [14, 22], 100, 105, 2, 20000, 18, 0.1, 5, 1],
            [1000, [1, 2, 3, 4, 5], 10, 9, 2, 1000, 3, 0.1, 1, 2],
        ],
    )
    def test_update_metrics(
        self,
        agent,
        steps,
        total_rewards,
        mean_reward,
        best_reward,
        plateau_count,
        expected_fps,
        expected_mean,
        learning_rate,
        interval,
        early_stop_count,
        capsys,
    ):
        """
        Ensure accuracy of the metrics displayed as the training
            progresses.
        Args:
            agent: OnPolicy/OffPolicy subclass.
            steps: Agent total environment steps so far.
            total_rewards: A list of finished episode rewards.
            mean_reward: Previous mean reward.
            best_reward: Previous best reward.
            plateau_count: Number of times agent reached plateau.
            expected_fps: Expected frame speed calculated given the other values.
            expected_mean: Expected calculated reward mean
            learning_rate: Start learning rate, before update.
            interval: Seconds back in time to set last reset time.
            early_stop_count: Number of times plateau reached its maximum
                patience and learning rate was reduced.
            capsys: _pytest.capture.CaptureFixture
        """
        agent = agent(**self.get_agent_kwargs(agent))
        agent.last_reset_time = perf_counter() - interval
        agent.steps = steps
        agent.total_rewards.extend(total_rewards)
        agent.mean_reward = mean_reward
        agent.best_reward = best_reward
        agent.plateau_count = plateau_count
        agent.early_stop_count = early_stop_count
        agent.divergence_monitoring_steps = 500000
        for model in agent.output_models:
            model.optimizer.learning_rate.assign(learning_rate)
        agent.update_metrics()
        cap = capsys.readouterr().out
        expected_learning_rate = learning_rate
        if plateau_count >= agent.plateau_reduce_patience:
            expected_learning_rate = agent.plateau_reduce_factor * learning_rate
        assert np.isclose(agent.frame_speed, expected_fps, rtol=3)
        assert agent.mean_reward == expected_mean
        assert np.isclose(agent.model.optimizer.learning_rate, expected_learning_rate)
        if mean_reward > best_reward:
            assert agent.early_stop_count == 0
            assert agent.plateau_count == 0
        if plateau_count >= agent.plateau_reduce_patience:
            assert agent.early_stop_count == early_stop_count + 1
        if learning_rate != expected_learning_rate:
            assert 'Learning rate reduced' in cap
            assert agent.plateau_count == 0
            assert agent.early_stop_count == early_stop_count + 1
        elif mean_reward < best_reward and steps >= agent.divergence_monitoring_steps:
            assert agent.plateau_count == plateau_count + 1

    def test_report_rewards(self, agent):
        """
        Test rewards are being reported as expected, and ensure
        unpromising branches are raising the appropriate exception.
        Args:
            agent: OnPolicy/OffPolicy subclass.
        """
        agent = agent(**self.get_agent_kwargs(agent))
        start_values = np.sort(np.random.randint(0, 100, 5))
        trial1 = optuna.trial.create_trial(
            params={'x': 2.0},
            distributions={'x': UniformDistribution(0, 10)},
            intermediate_values={i: start_values[i] for i in range(len(start_values))},
            value=55,
        )
        self.study.add_trial(trial1)
        trial2 = Trial(
            self.study, self.study._storage.create_new_trial(self.study._study_id)
        )
        agent.trial = trial2
        expected_rewards = start_values * 10
        for reward in expected_rewards:
            agent.total_rewards.append(reward)
            agent.report_rewards()
        reported_rewards = self.study.trials[-1].intermediate_values
        assert len(reported_rewards) == len(expected_rewards)
        for i in range(len(reported_rewards)):
            assert np.mean(expected_rewards[: i + 1]) == reported_rewards[i]
        trial3 = Trial(
            self.study, self.study._storage.create_new_trial(self.study._study_id)
        )
        agent.trial = trial3
        agent.total_rewards.clear()
        agent.reported_rewards = 0
        agent.total_rewards.append(-1)
        with pytest.raises(optuna.exceptions.TrialPruned):
            agent.report_rewards()

    def test_check_episodes(self, capsys, agent):
        """
        Ensure progress is displayed when expected and flags
            are updated afterwards.
        Args:
            capsys: _pytest.capture.CaptureFixture
            agent: OnPolicy/OffPolicy subclass.
        """
        agent = agent(**self.get_agent_kwargs(agent))
        agent.total_rewards.append(0)
        agent.training_start_time = perf_counter()
        agent.last_reset_time = perf_counter()
        agent.check_episodes()
        assert not capsys.readouterr().out
        agent.done_envs += len(self.envs)
        agent.check_episodes()
        self.assert_progress_displayed(capsys.readouterr().out)
        assert not agent.done_envs

    @pytest.mark.parametrize(
        'expected, mean_reward, steps, target_reward, max_steps, early_stop_count',
        [
            (False, 10, 0, 20, None, 0),
            (False, 10, 10, 11, 11, 0),
            (True, 100, 1000, 90, 2000, 0),
            (True, 200, 120, 500, 100, 0),
            (True, 1, 1, 1, 1, 0),
            (True, 100, 600000, 200, None, 5),
        ],
    )
    def test_training_done(
        self,
        capfd,
        expected,
        mean_reward,
        steps,
        target_reward,
        max_steps,
        early_stop_count,
        agent,
    ):
        """
        Test training status (done / not done) and ensure respective expected
            actions take place.
        Args:
            capfd: _pytest.capture.CaptureFixture
            expected: Expected training status where True indicates training
                should be done.
            mean_reward: Agent `mean_reward` attribute to set.
            steps: Agent `steps` attribute to set
            target_reward: Agent `target_reward` attribute to set.
            max_steps: Agent `max_steps` attribute to set
            early_stop_count: Number of times plateau reached its maximum
                patience and learning rate was reduced.
            agent: OnPolicy/OffPolicy subclass.
        """
        agent = agent(**self.get_agent_kwargs(agent))
        agent.mean_reward = mean_reward
        agent.steps = steps
        agent.target_reward = target_reward
        agent.max_steps = max_steps
        agent.early_stop_count = early_stop_count
        status = agent.training_done()
        messages = ['Reward achieved in', 'Maximum steps exceeded', 'Early stopping']
        if not expected:
            assert not status and not capfd.readouterr().out
        else:
            assert status
            cap = capfd.readouterr().out
            assert any([message in cap for message in messages])

    def test_concat_buffer_samples(self, off_policy_agent):
        """
        Test multiple buffer samples are concatenated properly.
        Args:
            off_policy_agent: OffPolicy subclass.
        """
        buffers = [ReplayBuffer1(10, batch_size=1) for _ in range(len(self.envs))]
        agent = off_policy_agent(
            **self.get_agent_kwargs(off_policy_agent, buffers=buffers)
        )
        with pytest.raises(ValueError, match=r'Sample larger than population'):
            agent.concat_buffer_samples()
        seen = []
        agent.batch_dtypes = ['float64', 'float16', 'float32', 'int16', 'int32']
        for i, buffer in enumerate(agent.buffers):
            observation = [[i], [i * 10], [i * 100], [i * 1000], [i * 10000]]
            seen.append(observation)
            buffer.append(*observation)
        expected = np.squeeze(np.array(seen)).T
        result = np.array(agent.concat_buffer_samples())
        assert [
            item.dtype for item in agent.concat_buffer_samples()
        ] == agent.batch_dtypes
        assert (result == expected).all()

    @pytest.mark.parametrize(
        'get_observation, store_in_buffers',
        [[False, False], [True, False], [False, True], [True, True]],
    )
    def test_step_envs(self, get_observation, store_in_buffers, agent):
        """
        Test observations are returned / stored in buffers when this is expected.
        Args:
            get_observation: arg passed to step_envs()
            store_in_buffers: arg passed to step_envs()
            agent: OnPolicy/OffPolicy subclass.
        """
        for buffer in self.buffers:
            buffer.main_buffer.clear()
            buffer.current_size = 0
        agent = agent(**self.get_agent_kwargs(agent))
        actions = [env.action_space.sample() for env in agent.envs]
        observations = agent.step_envs(actions, get_observation, store_in_buffers)
        if hasattr(agent, 'buffers'):
            if store_in_buffers:
                assert all([buffer.current_size > 0 for buffer in agent.buffers])
            else:
                assert all([buffer.current_size == 0 for buffer in agent.buffers])
        if get_observation:
            assert observations
            assert len(set([item.shape for item in observations[0]])) == 1
        else:
            assert not observations
        assert agent.steps == len(self.envs)

    @pytest.mark.skipif(not get_wandb_key(), reason='Wandb api key not available')
    @pytest.mark.parametrize(
        'target_reward, max_steps, monitor_session',
        [[18, None, None], [22, 55, None], [None, 100, 'test_session']],
    )
    def test_init_training(
        self, agent, capsys, target_reward, max_steps, monitor_session, tmpdir
    ):
        """
        Test training initialization and ensure training flags are set and
        a monitoring session is registered if one is specified.
        Args:
            agent: OnPolicy/OffPolicy subclass.
            capsys: _pytest.capture.CaptureFixture
            target_reward: Reward when reached, training should stop.
            max_steps: Number of agent steps, if reached, training should stop.
            monitor_session: wandb session name.
            tmpdir: py._path.local.LocalPath
        """
        checkpoints = [f'test-{i}.tf' for i in range(self.model_counts[agent])]
        agent = agent(**self.get_agent_kwargs(agent))
        agent.checkpoints = checkpoints
        with tmpdir.as_cwd():
            agent.init_training(target_reward, max_steps, monitor_session)
        assert agent.target_reward == target_reward
        assert agent.max_steps == max_steps
        assert agent.last_reset_time
        assert agent.training_start_time
        if monitor_session:
            assert wandb.run.name == monitor_session
            capsys.readouterr()

    def test_train_step(self, base_agent):
        """
        Ensure an exception is raised when train_step() abstract method is called.
        Args:
            base_agent: A base agent class.
        """
        agent = base_agent(**self.get_agent_kwargs(base_agent))
        with pytest.raises(NotImplementedError, match=r'should be implemented by'):
            agent.train_step()

    def test_get_model_outputs(self, base_agent):
        """
        Test single and multiple model outputs.
        Args:
            base_agent: A base agent class.
        """
        inputs = np.random.random((10, *self.envs[0].observation_space.shape))
        agent = base_agent(**self.get_agent_kwargs(base_agent))
        single_model_result = agent.get_model_outputs(inputs, self.model)
        multi_model_result = agent.get_model_outputs(inputs, [self.model, self.model])
        assert single_model_result.shape == multi_model_result[0].shape
        assert np.isclose(
            single_model_result.numpy(), multi_model_result[0].numpy()
        ).all()

    def test_play(
        self,
        agent,
        capsys,
        tmp_path,
        max_steps=10,
    ):
        """
        Play 1 game, and ensure resulting video / frames are saved.
        Args:
            agent: OnPolicy/OffPolicy subclass.
            capsys: _pytest.capture.CaptureFixture
            tmp_path: pathlib.PosixPath
            max_steps: Number of agent steps, if reached, training should stop.
        """
        agent_kwargs = {}
        agent_id = agent.__module__.split('.')[1]
        envs = self.envs if agent_id not in ['td3', 'ddpg'] else self.envs2
        agent_kwargs['envs'] = envs
        models = create_models(xagents.agents[agent_id], envs[0], agent_id)
        agent_kwargs.update(models)
        if issubclass(agent, OffPolicy) or agent == ACER:
            agent_kwargs['buffers'] = create_buffers(agent_id, 1000, 32, len(envs))
        agent = agent(**agent_kwargs)
        video_dir = tmp_path / agent_id / 'video'
        frame_dir = tmp_path / agent_id / 'frames'
        agent.play(video_dir, False, frame_dir, 0, max_steps)
        expected_frames = [frame_dir / f'{i:05d}.jpg' for i in range(max_steps)]
        for expected_frame in expected_frames:
            assert expected_frame.exists()
        assert len([*video_dir.rglob('openai*.mp4')])
        assert 'Maximum steps' in capsys.readouterr().out

    def test_fit(self, capsys):
        """
        Test agent.fit() and ensure the proper order of execution.
        Args:
            capsys: _pytest.capture.CaptureFixture
        """
        agent = DummyAgent(self.envs, self.model)
        agent.fit(max_steps=1)
        displayed = capsys.readouterr().out
        assert displayed.strip().split('\n') == [
            'step starting',
            'train_step',
            'step ending',
            'Maximum steps exceeded',
        ]

    @pytest.mark.parametrize('supported_agent', [DQN, TD3, DDPG])
    def test_fill_buffers(self, supported_agent, capsys):
        """
        Test buffers are filled up to the correct size, and
        filling progress was displayed.
        Args:
            supported_agent: An OffPolicy subclass.
            capsys: _pytest.capture.CaptureFixture
        """
        initial_size = 100
        max_size = 500
        n_buffers = len(self.envs)
        buffers = [
            ReplayBuffer1(500, initial_size=initial_size) for _ in range(n_buffers)
        ]
        agent = supported_agent(
            **self.get_agent_kwargs(supported_agent, buffers=buffers)
        )
        agent.fill_buffers()
        for i, buffer in enumerate(buffers):
            assert (
                buffer.current_size
                == buffer.initial_size
                == len(buffer.main_buffer)
                == initial_size
            )
            assert buffer.size == buffer.main_buffer.maxlen == max_size
        assert f'{n_buffers}/{n_buffers}' in capsys.readouterr().out
