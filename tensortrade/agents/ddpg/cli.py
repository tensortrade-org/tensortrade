cli_args = {
    'actor-model': {'help': 'Path to actor model .cfg file'},
    'critic-model': {'help': 'Path to critic model .cfg file'},
    'gradient-steps': {
        'help': 'Number of iterations per train step',
        'type': int,
        'hp_type': 'int',
    },
    'tau': {
        'help': 'Value used for syncing target model weights',
        'type': float,
        'default': 0.005,
        'hp_type': 'log_uniform',
    },
    'step-noise-coef': {
        'help': 'Coefficient multiplied by noise added to actions to step',
        'type': float,
        'default': 0.1,
        'hp_type': 'log_uniform',
    },
}
