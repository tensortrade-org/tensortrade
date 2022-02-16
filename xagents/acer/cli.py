from xagents import a2c

a2c_args = a2c.cli_args
acer_args = {
    'model': {'help': 'Path to model .cfg file'},
    'ema-alpha': {
        'help': 'Moving average decay passed to tf.train.ExponentialMovingAverage()',
        'type': float,
        'default': 0.99,
        'hp_type': 'log_uniform',
    },
    'replay-ratio': {
        'help': 'Lam value passed to np.random.poisson()',
        'type': int,
        'default': 4,
        'hp_type': 'categorical',
    },
    'epsilon': {
        'help': 'epsilon used in gradient updates',
        'type': float,
        'default': 1e-6,
        'hp_type': 'log_uniform',
    },
    'importance-c': {
        'help': 'Importance weight truncation parameter.',
        'type': float,
        'default': 10.0,
        'hp_type': 'log_uniform',
    },
    'delta': {
        'help': 'delta param used for trust region update',
        'type': float,
        'default': 1,
        'hp_type': 'log_uniform',
    },
    'trust-region': {
        'help': 'True by default, if this flag is specified,\n'
        'trust region updates will be used',
        'action': 'store_true',
    },
    'n-steps': {
        'help': 'Transition steps',
        'type': int,
        'default': 20,
        'hp_type': 'categorical',
    },
    'grad-norm': {
        'help': 'Gradient clipping value passed to tf.clip_by_value()',
        'type': float,
        'default': 10,
        'hp_type': 'log_uniform',
    },
}
cli_args = a2c_args.copy()
cli_args.update(acer_args)
