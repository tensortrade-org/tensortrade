from . import a2c, acer, ddpg, dqn, ppo, td3, trpo
from .a2c.agent import A2C
from .acer.agent import ACER
from .base import OffPolicy
from .ddpg.agent import DDPG
from .dqn.agent import DQN
from .ppo.agent import PPO
from .td3.agent import TD3
from .trpo.agent import TRPO
from .utils.cli import play_args, train_args, tune_args
from .utils.common import register_models

__author__ = 'alternativebug'
__email__ = 'alternativebug@outlook.com'
__license__ = 'MIT'
__version__ = '1.0.1'

agents = {
    'a2c': {'module': a2c, 'agent': A2C},
    'acer': {'module': acer, 'agent': ACER},
    'dqn': {'module': dqn, 'agent': DQN},
    'ppo': {'module': ppo, 'agent': PPO},
    'td3': {'module': td3, 'agent': TD3},
    'trpo': {'module': trpo, 'agent': TRPO},
    'ddpg': {'module': ddpg, 'agent': DDPG},
}
register_models(agents)
commands = {
    'train': (train_args, 'fit', 'Train given an agent and environment'),
    'play': (
        play_args,
        'play',
        'Play a game given a trained agent and environment',
    ),
    'tune': (
        tune_args,
        '',
        'Tune hyperparameters given an agent, hyperparameter specs, and environment',
    ),
}
