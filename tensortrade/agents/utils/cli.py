non_agent_args = {
    'env': {'help': 'gym environment id', 'required': True},
    'n-envs': {
        'help': 'Number of environments to create',
        'default': 1,
        'type': int,
        'hp_type': 'categorical',
    },
    'preprocess': {
        'help': 'If specified, states will be treated as atari frames\n'
        'and preprocessed accordingly',
        'action': 'store_true',
    },
    'lr': {
        'help': 'Learning rate passed to a tensorflow.keras.optimizers.Optimizer',
        'type': float,
        'default': 7e-4,
        'hp_type': 'log_uniform',
    },
    'opt-epsilon': {
        'help': 'Epsilon passed to a tensorflow.keras.optimizers.Optimizer',
        'type': float,
        'default': 1e-7,
        'hp_type': 'log_uniform',
    },
    'beta1': {
        'help': 'Beta1 passed to a tensorflow.keras.optimizers.Optimizer',
        'type': float,
        'default': 0.9,
        'hp_type': 'log_uniform',
    },
    'beta2': {
        'help': 'Beta2 passed to a tensorflow.keras.optimizers.Optimizer',
        'type': float,
        'default': 0.999,
        'hp_type': 'log_uniform',
    },
    'weights': {
        'help': 'Path(s) to model(s) weight(s) to be loaded by agent output_models',
        'nargs': '+',
    },
    'max-frame': {
        'help': 'If specified, max & skip will be applied during preprocessing',
        'action': 'store_true',
        'hp_type': 'categorical',
    },
}

off_policy_args = {
    'buffer-max-size': {
        'help': 'Maximum replay buffer size',
        'type': int,
        'default': 10000,
        'hp_type': 'int',
    },
    'buffer-initial-size': {
        'help': 'Replay buffer initial size',
        'type': int,
        'hp_type': 'int',
    },
    'buffer-batch-size': {
        'help': 'Replay buffer batch size',
        'type': int,
        'default': 32,
        'hp_type': 'categorical',
    },
}

agent_args = {
    'reward-buffer-size': {
        'help': 'Size of the total reward buffer, used for calculating\n'
        'mean reward value to be displayed.',
        'default': 100,
        'type': int,
    },
    'gamma': {
        'help': 'Discount factor',
        'default': 0.99,
        'type': float,
        'hp_type': 'log_uniform',
    },
    'display-precision': {
        'help': 'Number of decimals to be displayed',
        'default': 2,
        'type': int,
    },
    'seed': {'help': 'Random seed', 'type': int},
    'log-frequency': {'help': 'Log progress every n games', 'type': int},
    'checkpoints': {
        'help': 'Path(s) to new model(s) to which checkpoint(s) will be saved during training',
        'nargs': '+',
    },
    'history-checkpoint': {'help': 'Path to .parquet file to save training history'},
    'plateau-reduce-factor': {
        'help': 'Factor multiplied by current learning rate ' 'when there is a plateau',
        'type': float,
        'default': 0.9,
    },
    'plateau-reduce-patience': {
        'help': 'Minimum non-improvements to reduce lr',
        'type': int,
        'default': 10,
    },
    'early-stop-patience': {
        'help': 'Minimum plateau reduces to stop training',
        'type': int,
        'default': 3,
    },
    'divergence-monitoring-steps': {
        'help': 'Steps after which, plateau and early stopping are active',
        'type': int,
    },
    'quiet': {
        'help': 'If specified, no messages by the agent will be displayed'
        '\nto the console',
        'action': 'store_true',
    },
}

train_args = {
    'target-reward': {
        'help': 'Target reward when reached, training is stopped',
        'type': int,
    },
    'max-steps': {
        'help': 'Maximum number of environment steps, when reached, training is stopped',
        'type': int,
    },
    'monitor-session': {'help': 'Wandb session name'},
}

play_args = {
    'video-dir': {'help': 'Path to directory to save the resulting gameplay video'},
    'render': {
        'help': 'If specified, the gameplay will be rendered',
        'action': 'store_true',
    },
    'frame-dir': {'help': 'Path to directory to save game frames'},
    'frame-delay': {
        'help': 'Delay between rendered frames',
        'type': float,
        'default': 0,
    },
    'action-idx': {
        'help': 'Index of action output by agent.model',
        'type': int,
        'default': 0,
    },
    'frame-frequency': {
        'help': 'If --frame-dir is specified, save frames every n frames.',
        'type': int,
        'default': 1,
    },
}

tune_args = {
    'trial-steps': {
        'help': 'Maximum steps for a trial',
        'type': int,
        'default': 500000,
    },
    'n-trials': {'help': 'Number of trials to run', 'type': int, 'default': 1},
    'study': {'help': 'Name of optuna study'},
    'storage': {'help': 'Database url'},
    'n-jobs': {'help': 'Parallel trials', 'type': int, 'default': 1},
    'warmup-trials': {
        'help': 'warmup trials before pruning starts',
        'type': int,
        'default': 5,
    },
    'non-silent': {
        'help': 'tensorflow, optuna and agent are silenced at trial start\n'
        'to avoid repetitive import messages at each trial start, unless\n'
        'this flag is specified',
        'action': 'store_true',
    },
}
