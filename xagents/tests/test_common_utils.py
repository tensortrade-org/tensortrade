import os
import random
import uuid
from pathlib import Path

import gym
import pytest
from xagents.utils.common import AtariWrapper, create_buffers, get_wandb_key


@pytest.mark.parametrize(
    'resize_shape',
    [[random.randint(50, 100)] * 2 for _ in range(5)],
)
def test_atari_wrapper(resize_shape):
    """
    Test atari frame preprocessing, ensure shape and values are as expected.
    Args:
        resize_shape: (m, n) output frame size.
    """
    env = gym.make('PongNoFrameskip-v4')
    env = AtariWrapper(env, resize_shape=resize_shape)
    reset_state = env.reset()
    state, *_ = env.step(env.action_space.sample())
    assert state.shape == reset_state.shape == (*resize_shape, 1)


def test_get_wandb_key(tmp_path):
    """
    Test wandb API key is fetched correctly.
    Args:
        tmp_path: pathlib.PosixPath

    Returns:
        Wandb API key if found or None
    """
    env_var = 'WANDB_API_KEY'
    if not os.environ.get(env_var) and not (Path.home() / '.netrc').exists():
        assert not get_wandb_key()
    expected_key = uuid.uuid4().hex
    cfg_file = tmp_path / '.netrc'
    with open(cfg_file, 'w') as dummy_cfg:
        dummy_cfg.write(f'login user\npassword {expected_key}')
    assert get_wandb_key(cfg_file) == expected_key
    non_existent = tmp_path / 'non-existent'
    os.environ[env_var] = expected_key
    assert get_wandb_key(non_existent) == expected_key
    del os.environ[env_var]


@pytest.mark.parametrize(
    'max_size, batch_size, n_envs, initial_size, as_total',
    [
        [10000, 32, 16, None, True],
        [200000, 16, 16, 10000, False],
        [100000, 64, 2, None, False],
        [50000, 8, 2, 1000, True],
        [50000, 8, 3, 1000, False],
        [273645, 12, 1, 1, True],
    ],
)
def test_create_buffers(max_size, batch_size, n_envs, initial_size, as_total, agent_id):
    """
    Assert size parameters and buffer types are correct.
    Args:
        max_size: Buffer max size.
        batch_size: Buffer batch size when get_sample() is called.
        n_envs: Number of environments which will result in an equal
            number of buffers.
        initial_size: Buffer initial pre-training fill size.
        as_total: If False, total buffer initial, buffer max, and batch sizes
            are respectively buffer initial x n_envs, buffer max x n_envs,
            and batch_size x n_envs.
        agent_id: str, one of the keys in xagents.agents
    """
    expected_max_size = max_size
    expected_initial_size = initial_size or expected_max_size
    expected_batch_size = batch_size
    if as_total:
        expected_max_size //= n_envs
        expected_initial_size //= n_envs
        expected_batch_size //= n_envs
    if agent_id == 'acer':
        expected_batch_size = 1
    buffers = create_buffers(
        agent_id,
        max_size,
        batch_size,
        n_envs,
        initial_size,
        as_total,
    )
    expected_sizes = expected_max_size, expected_initial_size, expected_batch_size
    actual_sizes = buffers[0].size, buffers[0].initial_size, buffers[0].batch_size
    assert expected_sizes == actual_sizes
