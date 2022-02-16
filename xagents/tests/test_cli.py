import argparse
import random
import string

import pytest
import tensorflow as tf
import xagents
from gym.spaces import Discrete
from xagents.tests.utils import assert_flags_displayed, get_expected_flags
from xagents.utils.common import create_model


@pytest.mark.usefixtures('executor', 'envs', 'envs2')
class TestExecutor:
    """
    Tests for command line options.
    """

    def test_display_section(self, section, capsys):
        """
        Check if appropriate flags are displayed in help menu.
        Args:
            section: Tuple of argument group name and respective dict of args.
            capsys: _pytest.capture.CaptureFixture
        """
        self.executor.display_section(*section)
        cap = capsys.readouterr().out
        assert_flags_displayed(cap, *section)

    @staticmethod
    def assert_base_displayed(cap):
        """
        Assert base help menu is displayed.
        Args:
            cap: Text displayed to the console.

        Returns:
            None
        """
        for keyword in [
            'xagents',
            'Usage',
            'xagents <command> <agent> [options] [args]',
            'Available commands:',
        ]:
            assert keyword in cap

    def test_display_commands(self, section, capsys):
        """
        Test basic and extended help display.
        Args:
            section: Tuple of argument group name and respective dict of args.
            capsys: _pytest.capture.CaptureFixture
        """
        self.executor.display_commands({section[0]: section[1]})
        cap = capsys.readouterr().out
        self.assert_base_displayed(cap)
        if section:
            assert_flags_displayed(cap, *section)

    @staticmethod
    def add_arg_value(_action, _type, _nargs, hp_type, test_args, values, flag, tuning):
        """
        Generate random values for the given arg according to its attributes.
        Args:
            _action: If True, True will be stored.
            _type: Argument type.
            _nargs: Argument number of expected values.
            hp_type: hyperparameter type.
            test_args: argv that is being created.
            values: A dictionary of flags - expected values
            flag: Flag to which a value is generated.
            tuning: If True, flags that have an `hp_type`
                will support multiple parameters which will
                be parsed according to their type.

        Returns:
            None
        """
        multiple_values = [random.randint(0, 100000) for _ in range(10)]
        multiple_values.sort()
        if not _action:
            if tuning and hp_type:
                if hp_type == 'categorical':
                    value = multiple_values
                else:
                    value = multiple_values[:2]
            elif _nargs:
                values_type = _type or str
                value = [values_type(item) for item in multiple_values]
            elif _type in [int, float]:
                value = random.randint(0, 100)
            else:
                value = ''.join(random.sample(string.ascii_lowercase, 10))
            if not isinstance(value, list):
                test_args.append(str(value))
                values[flag] = value
            else:
                for item in value:
                    test_args.append(f'{item}')
                values[flag] = value
        else:
            values[flag] = True

    @pytest.mark.parametrize('tuning', [True, False])
    def test_add_args(self, section, tuning):
        """
        Add given section to executor args, generate values, parse args
        and test if the values match.
        Args:
            section: Tuple of argument group name and respective dict of args.
        """
        parser = argparse.ArgumentParser()
        self.executor.add_args(section[1], parser, tuning)
        test_args = []
        values = {}
        for flag, options in section[1].items():
            _help = options.get('help')
            _default = options.get('default')
            _type = options.get('type')
            _action = options.get('action')
            _nargs = options.get('nargs')
            _hp_type = options.get('hp_type')
            test_args.append(f'--{flag}')
            self.add_arg_value(
                _action, _type, _nargs, _hp_type, test_args, values, flag, tuning
            )
        parsed_args = parser.parse_args(test_args)
        for attr, value in values.items():
            assert getattr(parsed_args, attr.replace('-', '_')) == value

    def test_maybe_create_agent_base_display(self, capsys):
        """
        Ensure only base help menu is displayed.
        Args:
            capsys: _pytest.capture.CaptureFixture
        """
        self.executor.maybe_create_agent([])
        cap = capsys.readouterr().out
        assert 'flag' not in cap
        self.assert_base_displayed(cap)

    def test_maybe_create_agent_invalid_command(self):
        """
        Ensure exception is raised for a valid command.
        """
        with pytest.raises(AssertionError, match=r'Invalid command'):
            self.executor.maybe_create_agent(['invalid'])

    def test_maybe_create_agent_command_agent(self, command, agent_id, capsys):
        """
        Ensure command only / command + agent flags help is displayed accordingly.
        Args:
            command: One of the commands available in xagents.commands
            agent_id: One of the agent ids available in xagents.agents
            capsys: _pytest.capture.CaptureFixture
        """
        test_args = [[command], [command, agent_id]]
        for argv in test_args:
            expected_flags = get_expected_flags(argv)
            self.executor.maybe_create_agent(argv)
            cap = capsys.readouterr().out
            for flag in expected_flags:
                assert f'--{flag}' in cap

    def test_maybe_create_agent_no_display(self, command, agent_id, capsys):
        """
        Ensure nothing is displayed if the help menu is not invoked using
        relevant args.
        Args:
            command: One of the commands available in xagents.commands
            agent_id: One of the agent ids available in xagents.agents
            capsys: _pytest.capture.CaptureFixture
        """
        argv = [command, agent_id, '--env', 'test-env']
        self.executor.maybe_create_agent(argv)
        assert not capsys.readouterr().out
        assert self.executor.agent_id == agent_id
        assert self.executor.command == command

    @pytest.mark.parametrize('tuning', [True, False])
    def test_parse_known_args(self, command, agent_id, tuning):
        """
        Ensure agent kwargs + non-agent kwargs + command kwargs match
        the expected ones given command and agent + minimum other flags.
        Args:
            command: One of the commands available in xagents.commands
            agent_id: One of the agent ids available in xagents.agents
            tuning: If True, flags that have an `hp_type`
                will support multiple parameters which will
                be parsed according to their type.
        """
        self.executor.command = command
        self.executor.agent_id = agent_id
        argv = [command, agent_id, '--env', 'test-env']
        if command == 'train':
            with pytest.raises(AssertionError, match=r'train requires'):
                self.executor.parse_known_args(argv, tuning)
            argv.extend(['--target-reward', '18'])
        agent_args, non_agent_args, command_args = self.executor.parse_known_args(
            argv, tuning
        )
        unknown_argv = argv + ['--unknown-flag', 'unknown-value']
        with pytest.warns(UserWarning, match=r'Got unknown'):
            self.executor.parse_known_args(unknown_argv, tuning)
        actual = {**vars(agent_args), **vars(non_agent_args), **vars(command_args)}
        assert set(get_expected_flags(argv, True)) == set(actual.keys())

    def test_create_models(self, agent_id):
        """
        Test creation of models and ensure resulting output units match the expected.
        Args:
            agent_id: One of the agent ids available in xagents.agents
        """
        envs = self.envs if agent_id not in ['td3', 'ddpg'] else self.envs2
        expected_units = [
            envs[0].action_space.n
            if isinstance(envs[0].action_space, Discrete)
            else envs[0].action_space.shape[0]
        ]
        if agent_id == 'acer':
            expected_units.append(expected_units[-1])
        elif agent_id != 'dqn':
            expected_units.append(1)
        actual_units = []
        for model_type in ['model', 'actor_model', 'critic_model']:
            if model_type in xagents.agents[agent_id]:
                model = create_model(envs[0], agent_id, model_type)
                total_units = 2 if model_type == 'model' and agent_id != 'dqn' else 1
                output_units = [
                    layer.units
                    for layer in model.layers
                    if isinstance(layer, tf.keras.layers.Dense)
                ][-total_units:]
                actual_units.extend(output_units)
        assert expected_units == actual_units

    @pytest.mark.parametrize(
        'train_args',
        [
            {
                'args': 'train a2c --env PongNoFrameskip-v4 --entropy-coef 5 '
                '--value-loss-coef 33 --grad-norm 7 --checkpoints xyz.tf '
                '--reward-buffer-size 150 --n-steps 100 --gamma 0.88 '
                '--display-precision 3 --seed 55 '
                '--log-frequency 28 --n-envs 25 --lr 0.555 --opt-epsilon 0.3 '
                '--beta1 15 --beta2 12 --max-steps 1',
                'agent': {
                    'entropy_coef': 5,
                    'value_loss_coef': 33,
                    'grad_norm': 7,
                    'checkpoints': ['xyz.tf'],
                    'n_steps': 100,
                    'gamma': 0.88,
                    'display_precision': 3,
                    'seed': 55,
                    'log_frequency': 28,
                    'n_envs': 25,
                },
                'non_agent': {
                    'env': 'PongNoFrameskip-v4',
                    'agent': xagents.A2C,
                    'lr': 0.555,
                    'opt_epsilon': 0.3,
                    'beta1': 15,
                    'beta2': 12,
                },
            }
        ],
    )
    def test_train(self, capsys, train_args):
        """
        Do 1 train step and test explicit parameters.
        Args:
            capsys: _pytest.capture.CaptureFixture
            train_args: A dictionary of agent, non-agent and command args
                to be run by executor.
        """
        self.executor.execute(train_args['args'].split())
        assert 'Maximum steps exceeded' in capsys.readouterr().out
        for attr, value in train_args['agent'].items():
            assert getattr(self.executor.agent, attr) == value
        assert isinstance(self.executor.agent, train_args['non_agent']['agent'])
        assert (
            self.executor.agent.envs[0].unwrapped.spec.id
            == train_args['non_agent']['env']
        )
        assert (
            self.executor.agent.model.optimizer.learning_rate
            == train_args['non_agent']['lr']
        )
        assert (
            self.executor.agent.model.optimizer.epsilon
            == train_args['non_agent']['opt_epsilon']
        )
        assert (
            self.executor.agent.model.optimizer.beta_1
            == train_args['non_agent']['beta1']
        )
        assert (
            self.executor.agent.model.optimizer.beta_2
            == train_args['non_agent']['beta2']
        )
