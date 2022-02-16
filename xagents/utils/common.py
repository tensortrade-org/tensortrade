import configparser
import os
import re
from collections import deque
from pathlib import Path

import cv2
import gym
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xagents
from gym.spaces import Box, Discrete
from matplotlib import pyplot as plt
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from xagents.utils.buffers import ReplayBuffer1, ReplayBuffer2


class LazyFrames:
    """
    Efficient atari frame wrapper.
    """

    def __init__(self, frames):
        """
        Wrap frames.
        Args:
            frames: numpy array of atari frames.
        """
        self.frames = frames
        self.out = None
        self.dtype = frames.dtype
        self.shape = frames.shape

    def process_frame(self):
        """
        Get wrapped frames as numpy array.
        Returns:
            frames as numpy array
        """
        if self.out is None:
            self.out = np.array(self.frames)
            self.frames = None
        return self.out

    def __array__(self, dtype=None):
        out = self.process_frame()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.process_frame())

    def __getitem__(self, i):
        return self.process_frame()[i]

    def count(self):
        frames = self.process_frame()
        return frames.shape[frames.ndim - 1]


class AtariWrapper(gym.Wrapper):
    """
    gym wrapper for preprocessing atari frames.
    """

    def __init__(
        self,
        env,
        frame_skips=4,
        resize_shape=(84, 84),
        max_frame=False,
    ):
        """
        Initialize preprocessing settings.
        Args:
            env: gym environment that returns states as atari frames.
            frame_skips: Number of frame skips to use per environment step.
            resize_shape: (m, n) output frame size.
            max_frame: If True, max and skip is applied.
        """
        assert frame_skips > 1, 'frame_skips must be >= 1'
        super(AtariWrapper, self).__init__(env)
        self.skips = frame_skips
        self.frame_shape = resize_shape
        self.observation_space._shape = (*resize_shape, 1)
        self.max_frame = max_frame
        self.frame_buffer = deque(maxlen=2)

    def process_frame(self, frame):
        """
        Resize and convert atari frame to grayscale.
        Args:
            frame: Atari frame as numpy.ndarray

        Returns:
            Processed frame.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, self.frame_shape)
        return LazyFrames(np.expand_dims(frame, -1))

    def step(self, action):
        """
        Step respective to self.skips.
        Args:
            action: Action supported by self.env

        Returns:
            (state, reward, done, info)
        """
        total_reward = 0
        state, done, info = 3 * [None]
        for _ in range(self.skips):
            state, reward, done, info = self.env.step(action)
            if self.max_frame:
                self.frame_buffer.append(state)
                state = np.max(np.stack(self.frame_buffer), axis=0)
            total_reward += reward
            if done:
                break
        return self.process_frame(state), total_reward, done, info

    def reset(self, **kwargs):
        """
        Reset self.env
        Args:
            **kwargs: kwargs passed to self.env.reset()

        Returns:
            Processed atari frame.
        """
        state = self.env.reset(**kwargs)
        if self.max_frame:
            self.frame_buffer.clear()
            self.frame_buffer.append(state)
        return self.process_frame(state)


def create_envs(env_name, n=1, preprocess=True, *args, **kwargs):
    """
    Create gym environment and initialize preprocessing settings.
    Args:
        env_name: Name of the environment to be passed to gym.make()
        n: Number of environments to create.
        preprocess: If True, AtariWrapper will be used.
        *args: args to be passed to AtariWrapper
        **kwargs: kwargs to be passed to AtariWrapper

    Returns:
        A list of gym environments.
    """
    envs = [gym.make(env_name) for _ in range(n)]
    if preprocess:
        assert len(envs[0].observation_space.shape) == 3, (
            f'Cannot use AtariWrapper or --preprocess for non-atari environment '
            f'{envs[0].spec.id}, with input '
            f'shape {envs[0].observation_space.shape}'
        )
        envs = [AtariWrapper(env, *args, **kwargs) for env in envs]
    return envs


class ModelReader:
    """
    Model utility class to create basic keras models from configuration files.
    """

    def __init__(self, cfg_file, output_units, input_shape, optimizer=None, seed=None):
        """
        Initialize model parser.
        Args:
            cfg_file: Path to .cfg file having that will be created.
            output_units: A list of output units that must be of the
                same size as the number of dense layers in the configuration
                without specified units.
            input_shape: input shape passed to tf.keras.layers.Input()
            optimizer: tf.keras.optimizers.Optimizer with which the resulting
                model will be compiled.
            seed: Random seed used by layer initializers.
        """
        self.initializers = {'orthogonal': Orthogonal, 'glorot_uniform': GlorotUniform}
        self.cfg_file = cfg_file
        with open(cfg_file) as cfg:
            self.parser = configparser.ConfigParser()
            self.parser.read_file(cfg)
        self.optimizer = optimizer
        self.output_units = output_units
        self.input_shape = input_shape
        self.seed = seed
        self.output_count = 0

    def get_initializer(self, section):
        """
        Get layer initializer if specified in the configuration.
        Args:
            section: str, representing section unique name.

        Returns:
            tf.keras.initializers.Initializer
        """
        initializer_name = self.parser[section].get('initializer')
        gain = self.parser[section].get('gain')
        if self.seed is not None:
            initializer_name = initializer_name or 'glorot_uniform'
        initializer_kwargs = {'seed': self.seed}
        if gain:
            initializer_kwargs.update({'gain': float(gain)})
        initializer = self.initializers.get(initializer_name)
        if initializer:
            return initializer(**initializer_kwargs)

    def create_convolution(self, section):
        """
        Parse convolution layer parameters and create layer.
        Args:
            section: str, representing section unique name.

        Returns:
            tf.keras.layers.Conv1D
        """
        filters = int(self.parser[section]['filters'])
        kernel_size = int(self.parser[section]['size'])
        stride = int(self.parser[section]['stride'])
        activation = self.parser[section].get('activation')
        return Conv1D(
            filters,
            kernel_size,
            stride,
            activation=activation,
            kernel_initializer=self.get_initializer(section),
        )

    def create_dense(self, section):
        """
        Parse dense layer parameters and create layer.
        Args:
            section: str, representing section unique name.

        Returns:
            tf.keras.layers.Dense
        """
        units = self.parser[section].get('units')
        if not units:
            assert (
                len(self.output_units) > self.output_count
            ), 'Output units given are less than dense layers required'
            units = self.output_units[self.output_count]
            self.output_count += 1
        activation = self.parser[section].get('activation')
        return Dense(
            units, activation, kernel_initializer=self.get_initializer(section)
        )

    def build_model(self):
        """
        Parse all configuration sections, create respective layers, create and
        compile model.

        Returns:
            tf.keras.Model
        """
        outputs = []
        common_layer = None
        input_layer = current_layer = Input(self.input_shape)
        sections = self.parser.sections()
        assert sections, f'Empty model configuration {self.cfg_file}'
        for section in self.parser.sections():
            if section.startswith('convolutional'):
                current_layer = self.create_convolution(section)(current_layer)
            if section.startswith('flatten'):
                current_layer = Flatten()(current_layer)
            if section.startswith('dense'):
                current_layer = self.create_dense(section)(
                    common_layer if common_layer is not None else current_layer
                )
            if self.parser[section].get('common'):
                common_layer = current_layer
            if self.parser[section].get('output'):
                outputs.append(current_layer)
        self.output_count = 0
        model = Model(input_layer, outputs)
        if self.optimizer:
            model.compile(self.optimizer)
        return model


def allocate_by_network(available_cfg, cfg_group):
    """
    Allocate given cfg file path into given group's `cnn` or `ann`
    Args:
        available_cfg: Path to .cfg file in `models` configuration folder.
        cfg_group: A dictionary with `cnn` and `ann` as keys.

    Returns:
        None
    """
    if 'cnn' in available_cfg:
        cfg_group['cnn'].append(available_cfg)
    if 'ann' in available_cfg:
        cfg_group['ann'].append(available_cfg)


def register_models(agents):
    """
    Register default model configuration files found in all agent `models`
    configuration folders to be added to xagents.agents.
    Args:
        agents: xagents.agents

    Returns:
        None
    """
    for agent_data in agents.values():
        models_folder = Path(agent_data['module'].__file__).parent / 'models'
        available_cfgs = [model_cfg.as_posix() for model_cfg in models_folder.iterdir()]
        actor_cfgs = {'cnn': [], 'ann': []}
        critic_cfgs = {'cnn': [], 'ann': []}
        model_cfgs = {'cnn': [], 'ann': []}
        for available_cfg in available_cfgs:
            if 'actor' not in available_cfg and 'critic' not in available_cfg:
                allocate_by_network(available_cfg, model_cfgs)
            elif 'actor' in available_cfg and 'critic' in available_cfg:
                allocate_by_network(available_cfg, model_cfgs)
            elif 'actor' in available_cfg:
                allocate_by_network(available_cfg, actor_cfgs)
            elif 'critic' in available_cfg:
                allocate_by_network(available_cfg, critic_cfgs)
        for key, val in zip(
            ['actor_model', 'critic_model', 'model'],
            [actor_cfgs, critic_cfgs, model_cfgs],
        ):
            if any(val.values()):
                agent_data[key] = val


def get_wandb_key(configuration_file=None):
    """
    Check ~/.netrc and WANDB_API_KEY environment variable for wandb api key.
    Args:
        configuration_file: Path to wandb configuration file, if not specified,
            defaults to ~/.netrc

    Returns:
        Key found or None
    """
    login_file = (
        Path(configuration_file) if configuration_file else Path.home() / '.netrc'
    )
    if login_file.exists():
        with open(login_file) as cfg:
            contents = cfg.read()
            key = re.findall(r"([a-fA-F\d]{32,})", contents)
            if key:
                return key[0]
    return os.environ.get('WANDB_API_KEY')


def plot_history(
    paths,
    agents,
    env,
    plot='mean_reward',
    benchmark='step',
    history_interval=1,
    time_unit='hour',
):
    """
    Plot a single agent training history given a column name,
    OR
    plot multiple agent training histories compared in a single figure.
    Args:
        paths: List of paths to .parquet files that have the training histories.
        agents: List of agent ids for labels / legend.
        env: Environment id.
        plot: Name of the column to be compared.
        benchmark: `step` or `time`
        history_interval: Plot every n data points.
        time_unit: Assuming the `time` column has seconds, time
            will be divided accordingly.

    Returns:
        None
    """
    time_divisors = {'hour': 3600, 'minute': 60, 'second': 1}
    assert len(paths) == len(agents), (
        f'Expected `paths` and `agents` to have the same sizes, '
        f'got {len(paths)} vs {len(agents)}'
    )
    histories = [
        pd.read_parquet(path).sort_values('time').iloc[::history_interval]
        for path in paths
    ]
    for history in histories:
        data = history[benchmark]
        if benchmark == 'time':
            data = history[benchmark] / time_divisors[time_unit]
        plt.plot(data, history[plot])
    if len(agents) == 1:
        title = f'{agents[0]} - {env}'
    else:
        title = env
    plt.title(title)
    x_label = f'{benchmark} ({time_unit}s)' if benchmark == 'time' else benchmark
    plt.xlabel(x_label)
    plt.ylabel(plot.replace('_', ' '))
    plt.legend(agents, loc='lower right')
    plt.grid()


def write_from_dict(_dict, path):
    """
    Write to .parquet given a dict
    Args:
        _dict: Dictionary of label: [scalar]
        path: Path to .parquet fiile.

    Returns:
        None
    """
    table = pa.Table.from_pydict(_dict)
    pq.write_to_dataset(table, root_path=path, compression='gzip')


def create_model(
    env, agent_id, model_type, optimizer_kwargs=None, seed=None, model_cfg=None
):
    """
    Create model from a given model cfg or from default configuration if any.
    Args:
        env: gym environment.
        agent_id: str, one of the keys in xagents.agents
        model_type: 'model' or 'actor_model' or 'critic_model'
        optimizer_kwargs: A dictionary of epsilon, that may contain one or more of
            `learning_rate`, `beta_1` or `beta_2`
        seed: random seed passed to layer initializers.
        model_cfg: Path to .cfg containing a compatible model configuration.

    Returns:
        tf.keras.Model
    """
    units = [
        env.action_space.n
        if isinstance(env.action_space, Discrete)
        else env.action_space.shape[0]
    ]
    if len(env.observation_space.shape) == 3:
        network_type = 'cnn'
    else:
        network_type = 'ann'
    try:
        model_cfg = model_cfg or xagents.agents[agent_id][model_type][network_type][0]
    except IndexError:
        model_cfg = None
    models_folder = Path(xagents.agents[agent_id]['module'].__file__).parent / 'models'
    assert model_cfg, (
        f'You should specify `model_cfg`. No default '
        f'{network_type.upper()} model found in\n{models_folder}'
    )
    if agent_id == 'acer':
        units.append(units[-1])
    elif 'actor' in model_cfg and 'critic' in model_cfg:
        units.append(1)
    elif 'critic' in model_cfg:
        units[0] = 1
    optimizer_kwargs = optimizer_kwargs or {}
    model_reader = ModelReader(
        model_cfg,
        units,
        env.observation_space.shape,
        Adam(**optimizer_kwargs),
        seed,
    )
    if agent_id in ['td3', 'ddpg'] and 'critic' in model_cfg:
        assert isinstance(env.action_space, Box), (
            f'Invalid environment: {env.spec.id}. {agent_id.upper()} supports '
            f'environments with a Box action space only, got {env.action_space}'
        )
        model_reader.input_shape = (
            model_reader.input_shape[0] + env.action_space.shape[0]
        )
    return model_reader.build_model()


def create_models(options, env, agent_id, **kwargs):
    """
    Create agent models
    Args:
        options: iterable that has `model` or (`actor_model` and `critic_model`)
        env: gym environment
        agent_id: str, one of the keys in xagents.agents
        **kwargs: kwargs passed to xagents.utils.common.create_model()

    Returns:
        list of model(s)
    """
    model_types = ['model', 'actor_model', 'critic_model']
    models = {}
    for model_type in model_types:
        if model_type in options:
            model_cfg = options[model_type]
            if not isinstance(model_cfg, (str, Path)):
                model_cfg = None
            models[model_type] = create_model(
                env, agent_id, model_type, model_cfg=model_cfg, **kwargs
            )
    return models


def create_buffers(
    agent_id,
    max_size,
    batch_size,
    n_envs,
    initial_size=None,
    as_total=True,
):
    """
    Create deque-based or numpy-based replay buffers.
    Args:
        agent_id: str, one of the keys in xagents.agents
        max_size: Buffer max size.
        batch_size: Buffer batch size when get_sample() is called.
        n_envs: Number of environments which will result in an equal
            number of buffers.
        initial_size: Buffer initial pre-training fill size.
        as_total: If False, total buffer initial, buffer max, and batch sizes
            are respectively buffer initial x n_envs, buffer max x n_envs,
            and batch_size x n_envs.

    Returns:
        list of buffers.
    """
    initial_size = initial_size or max_size
    if as_total:
        max_size //= n_envs
        initial_size //= n_envs
        batch_size //= n_envs
    if agent_id == 'acer':
        batch_size = 1
    if agent_id in ['td3', 'ddpg']:
        buffers = [
            ReplayBuffer2(
                max_size,
                5,
                initial_size=initial_size,
                batch_size=batch_size,
            )
            for _ in range(n_envs)
        ]
    else:
        buffers = [
            ReplayBuffer1(
                max_size,
                initial_size=initial_size,
                batch_size=batch_size,
            )
            for _ in range(n_envs)
        ]
    return buffers


def create_agent(agent_id, agent_kwargs, non_agent_kwargs, trial=None):
    """
    Create agent with all sub-components, including environments,
    models and buffers.
    Args:
        agent_id: str, one of the keys in xagents.agents
        agent_kwargs: dictionary of agent kwargs, values
        non_agent_kwargs: dictionary of non-agent kwargs, values
        trial: optuna.trial.Trial

    Returns:
        agent.
    """
    agent_kwargs['trial'] = trial
    envs = create_envs(
        non_agent_kwargs['env'],
        non_agent_kwargs['n_envs'],
        non_agent_kwargs['preprocess'],
        max_frame=non_agent_kwargs['max_frame'],
    )
    agent_kwargs['envs'] = envs
    optimizer_kwargs = {
        'learning_rate': non_agent_kwargs['lr'],
        'beta_1': non_agent_kwargs['beta1'],
        'beta_2': non_agent_kwargs['beta2'],
        'epsilon': non_agent_kwargs['opt_epsilon'],
    }
    models = create_models(
        agent_kwargs,
        envs[0],
        agent_id,
        optimizer_kwargs=optimizer_kwargs,
        seed=agent_kwargs['seed'],
    )
    agent_kwargs.update(models)
    if (
        issubclass(xagents.agents[agent_id]['agent'], xagents.OffPolicy)
        or agent_id == 'acer'
    ):
        buffers = create_buffers(
            agent_id,
            non_agent_kwargs['buffer_max_size'],
            non_agent_kwargs['buffer_batch_size'],
            non_agent_kwargs['n_envs'],
            non_agent_kwargs['buffer_initial_size'],
        )
        agent_kwargs['buffers'] = buffers
    agent = xagents.agents[agent_id]['agent'](**agent_kwargs)
    if non_agent_kwargs['weights']:
        n_weights = len(non_agent_kwargs['weights'])
        n_models = len(agent.output_models)
        assert (
            n_weights == n_models
        ), f'Expected {n_models} weights to load, got {n_weights}'
        for weight, model in zip(non_agent_kwargs['weights'], agent.output_models):
            model.load_weights(weight).expect_partial()
    return agent
