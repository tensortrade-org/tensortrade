import gym
import numpy as np
import optuna
import pytest
import tensortrade.agents as agents
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensortrade.agents import ACER, DDPG, DQN, TD3
from tensortrade.agents.base import BaseAgent, OffPolicy, OnPolicy
from tensortrade.agents.cli import Executor
from tensortrade.agents.utils.buffers import ReplayBuffer1, ReplayBuffer2
from tensortrade.agents.utils.cli import (agent_args, non_agent_args, play_args,
                               train_args, tune_args)

optuna.logging.set_verbosity(optuna.logging.ERROR)


@pytest.fixture(scope='function')
def executor(request):
    """
    Fixture for testing command line options.
    Args:
        request: _pytest.fixtures.SubRequest

    Returns:
        agents.cli.Executor
    """
    request.cls.executor = Executor()


@pytest.fixture(
    params=[
        ('non-agent', non_agent_args),
        ('agent', agent_args),
        ('train', train_args),
        ('play', play_args),
        ('tune', tune_args),
        ('do-nothing', {}),
    ]
)
def section(request):
    """
    Fixture for testing help menu and parsing sanity.
    Args:
        request: _pytest.fixtures.SubRequest

    Yields:
        Tuple of argument group name and respective dict of args.
    """
    yield request.param


#@pytest.fixture(scope='class')
#def envs(request):
#    """
#    Fixture of PongNoFrameskip-v4 environments used to test agents that support
#    environments with Discrete action space.
#    Args:
#        request: _pytest.fixtures.SubRequest
#
#    Returns:
#        A list of 4 pong environments.
#    """
#    request.cls.envs = [gym.make('PongNoFrameskip-v4') for _ in range(4)]


#@pytest.fixture(scope='class')
#def envs2(request):
#    """
#    Fixture of BipedalWalker-v3 environments used to test agents that support
#    environments with continuous action space.
#    Args:
#        request: _pytest.fixtures.SubRequest
#
#    Returns:
#        A list of 4 bipedal walker environments.
#    """
#    request.cls.envs2 = [gym.make('BipedalWalker-v3') for _ in range(4)]


#@pytest.fixture(scope='class')
#def envs2(request):
#    """
#    Fixture of TradingEnv-v0 environments used to test agents that support
#    environments with continuous action space.
#    Args:
#        request: _pytest.fixtures.SubRequest
#
#    Returns:
#        A list of 4 bipedal walker environments.
#    """
#
#    # ...
#
#    max_allowed_loss = 0.90
#    min_periods = window_size  # Minimum of window_size
#
#    observer = default.observers.TensorTradeObserver(
#        portfolio=portfolio,
#        feed=feed,
#        renderer_feed=renderer_feed,
#        window_size=window_size,
#        min_periods=min_periods
#    )
#
#    stopper = default.stoppers.MaxLossStopper(
#        max_allowed_loss=max_allowed_loss
#    )
#
#    informer = default.informers.TensorTradeInformer()
#
#    renderer = default.renderers.PlotlyTradingChart()
#
#    env = gym.envs.register(
#        id='TradingEnv-v0',
#        entry_point='tensortrade.env.generic.environment:TradingEnv',
#        kwargs={
#            'portfolio': portfolio,
#            'action_scheme': action_scheme,
#            'reward_scheme': reward_scheme,
#            'observer': observer,
#            'stopper': stopper,
#            'informer': informer,
#            'feed': feed,
#                'renderer_feed': renderer_feed,
#                'renderer': renderer,
#                'min_periods': min_periods,
#                'window_size': window_size,
#        },
#    )
#
#    request.cls.envs2 = [gym.make('TradingEnv-v0') for _ in range(4)]


@pytest.fixture(scope='class')
def model(request):
    """
    Fixture used by all agents in tests that require a model but do not
    require the agent default model. It should be included in
    pytest.mark.usefixtures() decorator to allow its availability as
    a class attribute.
    Args:
        request: _pytest.fixtures.SubRequest

    Returns:
        None
    """
    x0 = Input((30, 2))
    x = Dense(1, 'relu')(x0)
    x = Dense(1, 'relu')(x)
    x = Dense(1, 'relu')(x)
    model = request.cls.model = Model(x0, x)
    model.compile('adam')


@pytest.fixture(scope='class')
def buffers(request):
    """
    Fixture used by off-policy agent tests that require a replay buffer
    that is not size specific. It should be included in
    pytest.mark.usefixtures() decorator to allow its availability as
    a class attribute.
    Args:
        request: _pytest.fixtures.SubRequest

    Returns:
        None
    """
    request.cls.buffers = [ReplayBuffer1(200, batch_size=155) for _ in range(4)]


@pytest.fixture(scope='function')
def buffer1(request):
    """
    Fixture used to test ReplayBuffer1.
    Args:
        request: _pytest.fixtures.SubRequest

    Returns:
        None
    """
    request.cls.buffer = ReplayBuffer1(100, batch_size=4)


@pytest.fixture(scope='function')
def buffer2(request):
    """
    Fixture used to test ReplayBuffer2.
    Args:
        request: _pytest.fixtures.SubRequest

    Returns:
        None
    """
    request.cls.buffer = ReplayBuffer2(100, 5, batch_size=4)


@pytest.fixture(params=[item['agent'] for item in agents.agents.values()])
def agent(request):
    """
    Fixture with agents available in agents.agents
    Args:
        request: _pytest.fixtures.SubRequest

    Yields:
        OnPolicy/OffPolicy subclass.
    """
    yield request.param


@pytest.fixture(params=[agent_id for agent_id in agents.agents])
def agent_id(request):
    """
    Fixture with agent ids available in agents.agents
    Args:
        request: _pytest.fixtures.SubRequest

    Yields:
        Agent id as str
    """
    yield request.param


@pytest.fixture(params=[command for command in agents.commands])
def command(request):
    """
    Fixture with commands available in agents.commands
    Args:
        request: _pytest.fixtures.SubRequest

    Yields:
        Command as str
    """
    yield request.param


@pytest.fixture(params=[ACER, TD3, DQN, DDPG])
def off_policy_agent(request):
    """
    Fixture with off-policy agents.
    Args:
        request: _pytest.fixtures.SubRequest

    Yields:
        OffPolicy subclass.
    """
    yield request.param


@pytest.fixture(params=[BaseAgent, OnPolicy, OffPolicy])
def base_agent(request):
    """
    Fixture with agent base classes, used to test base and
    abstract methods.
    Args:
        request: _pytest.fixtures.SubRequest

    Yields:
        OnPolicy/OffPolicy subclass.
    """
    yield request.param


@pytest.fixture(params=[ReplayBuffer1, ReplayBuffer2])
def buffer_type(request):
    """
    Fixture with buffer types to test buffer attributes.
    Args:
        request: _pytest.fixtures.SubRequest

    Yields:
        BaseBuffer subclass.
    """
    yield request.param


@pytest.fixture(scope='class')
def observations(request):
    """
    Fixture to be used by class methods that require creation of a batch.
    Args:
        request: _pytest.fixtures.SubRequest

    Returns:
        None
    """
    states = np.random.randint(0, 255, (10, 84, 1))
    actions = np.random.randint(0, 6, 10)
    rewards = np.random.randint(0, 10, 10)
    dones = np.random.randint(0, 1000000, 10)
    new_states = np.random.randint(0, 255, (30, 2, 1))
    request.cls.observations = [
        [*items] for items in zip(states, actions, rewards, dones, new_states)
    ]


@pytest.fixture(scope='class')
def study(request):
    """
    Fixture that provides optuna.study.Study to test tuning components.
    Args:
        request: _pytest.fixtures.SubRequest

    Returns:
        None
    """
    pruner = optuna.pruners.MedianPruner(1)
    request.cls.study = optuna.create_study(pruner=pruner, direction='maximize')
