cli_args = {
    'model': {'help': 'Path to model .cfg file'},
    'entropy-coef': {
        'help': 'Entropy coefficient for loss calculation',
        'type': float,
        'default': 0.01,
        'hp_type': 'log_uniform',
    },
    'value-loss-coef': {
        'help': 'Value loss coefficient for value loss calculation',
        'type': float,
        'default': 0.5,
        'hp_type': 'log_uniform',
    },
    'grad-norm': {
        'help': 'Gradient clipping value passed to tf.clip_by_value()',
        'type': float,
        'default': 0.5,
        'hp_type': 'log_uniform',
    },
    'n-steps': {
        'help': 'Transition steps',
        'type': int,
        'default': 5,
        'hp_type': 'categorical',
    },
}
