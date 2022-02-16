import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import optuna
import tensorflow as tf
import xagents
from xagents.utils.cli import agent_args, non_agent_args, off_policy_args
from xagents.utils.common import create_agent


class Objective:
    """
    Objective function wrapper class.
    """

    def __init__(
        self, agent_id, agent_known_args, non_agent_known_args, command_known_args
    ):
        """
        Initialize objective function parameters.
        Args:
            agent_id: One of the agent ids available in xagents.agents
            agent_known_args: argparse.Namespace, containing options passed to agent.
            non_agent_known_args: argparse.Namespace, containing options passed to
                environments, optimizer and other non-agent objects.
            command_known_args: argparse.Namespace, containing command-specific options.
        """
        self.agent_id = agent_id
        self.agent_args = agent_known_args
        self.non_agent_args = non_agent_known_args
        self.command_args = command_known_args
        self.arg_groups = [
            (
                vars(agent_known_args),
                {**agent_args, **xagents.agents[agent_id]['module'].cli_args},
                self.agent_args,
            ),
            (
                vars(non_agent_known_args),
                {**non_agent_args, **off_policy_args},
                self.non_agent_args,
            ),
        ]

    def set_trial_values(self, trial):
        """
        Set values of all available hyperparameters that are specified
        to be tuned.
        Args:
            trial: optuna.trial.Trial

        Returns:
            None
        """
        for parsed_args, default_args, namespace in self.arg_groups:
            for arg, possible_values in parsed_args.items():
                hp_type = default_args[arg.replace('_', '-')].get('hp_type')
                trial_value = possible_values
                if isinstance(possible_values, list):
                    if hp_type and len(possible_values) == 1:
                        trial_value = possible_values[0]
                    elif hp_type == 'categorical':
                        trial_value = trial.suggest_categorical(arg, possible_values)
                    elif hp_type == 'log_uniform':
                        trial_value = trial.suggest_loguniform(arg, *possible_values)
                    elif hp_type == 'int':
                        trial_value = trial.suggest_int(arg, *possible_values)
                setattr(namespace, arg, trial_value)

    def __call__(self, trial):
        """
        Objective function called by optuna.study.Study.optimize
        Args:
            trial: optuna.trial.Trial

        Returns:
            Trial reward.
        """
        self.set_trial_values(trial)
        agent = create_agent(
            self.agent_id, vars(self.agent_args), vars(self.non_agent_args), trial
        )
        agent.fit(max_steps=self.command_args.trial_steps)
        trial_reward = np.around(np.mean(agent.total_rewards or [0]), 2)
        return trial_reward


def run_trial(
    agent_id,
    agent_known_args,
    non_agent_known_args,
    command_known_args,
):
    """
    Run one trial.
    Args:
        agent_id: One of the agent ids available in xagents.agents
        agent_known_args: argparse.Namespace, containing options passed to agent.
        non_agent_known_args: argparse.Namespace, containing options passed to
            environments, optimizer and other non-agent objects.
        command_known_args: argparse.Namespace, containing command-specific options.

    Returns:
        None
    """
    if not command_known_args.non_silent:
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        agent_known_args.quiet = True
    study = optuna.load_study(
        command_known_args.study, storage=command_known_args.storage
    )
    objective = Objective(
        agent_id, agent_known_args, non_agent_known_args, command_known_args
    )
    study.optimize(objective, n_trials=1)


def run_tuning(agent_id, agent_known_args, non_agent_known_args, command_known_args):
    """
    Run tuning session with tuning specs specified in command_known_args.
    Args:
        agent_id: One of the agent ids available in xagents.agents
        agent_known_args: argparse.Namespace, containing options passed to agent.
        non_agent_known_args: argparse.Namespace, containing options passed to
            environments, optimizer and other non-agent objects.
        command_known_args: argparse.Namespace, containing command-specific options.

    Returns:
        None
    """
    trial_kwargs = {
        'agent_id': agent_id,
        'agent_known_args': agent_known_args,
        'non_agent_known_args': non_agent_known_args,
        'command_known_args': command_known_args,
    }
    pruner = optuna.pruners.MedianPruner(command_known_args.warmup_trials)
    optuna.create_study(
        study_name=command_known_args.study,
        storage=command_known_args.storage,
        load_if_exists=True,
        direction='maximize',
        pruner=pruner,
    )
    for _ in range(command_known_args.n_trials // command_known_args.n_jobs):
        with ProcessPoolExecutor(command_known_args.n_jobs) as executor:
            future_trials = [
                executor.submit(run_trial, **trial_kwargs)
                for _ in range(command_known_args.n_jobs)
            ]
            for future_trial in as_completed(future_trials):
                future_trial.result()
