from xagents import a2c, ppo

a2c_args = a2c.cli_args
ppo_args = ppo.cli_args
trpo_args = {
    'actor-model': {'help': 'Path to actor model .cfg file'},
    'critic-model': {'help': 'Path to critic model .cfg file'},
    'max-kl': {
        'help': 'Maximum KL divergence used for calculating Lagrange multiplier',
        'type': float,
        'default': 1e-3,
        'hp_type': 'log_uniform',
    },
    'cg-iterations': {
        'help': 'Gradient conjugation iterations per train step',
        'type': int,
        'default': 10,
    },
    'cg-residual-tolerance': {
        'help': 'Gradient conjugation residual tolerance parameter',
        'type': float,
        'default': 1e-10,
        'hp_type': 'log_uniform',
    },
    'cg-damping': {
        'help': 'Gradient conjugation damping parameter',
        'type': float,
        'default': 1e-3,
        'hp_type': 'log_uniform',
    },
    'actor-iterations': {
        'help': 'Actor optimization iterations per train step',
        'type': int,
        'default': 10,
        'hp_type': 'int',
    },
    'critic-iterations': {
        'help': 'Critic optimization iterations per train step',
        'type': int,
        'default': 3,
        'hp_type': 'int',
    },
    'fvp-n-steps': {
        'help': 'Value used to skip every n-frames used to calculate FVP',
        'type': int,
        'default': 5,
        'hp_type': 'int',
    },
    'entropy-coef': {
        'help': 'Entropy coefficient for loss calculation',
        'type': float,
        'default': 0,
        'hp_type': 'log_uniform',
    },
    'lam': {
        'help': 'GAE-Lambda for advantage estimation',
        'type': float,
        'default': 1.0,
        'hp_type': 'log_uniform',
    },
    'n-steps': {
        'help': 'Transition steps',
        'type': int,
        'default': 512,
        'hp_type': 'categorical',
    },
}
cli_args = a2c_args.copy()
cli_args.update(ppo_args)
cli_args.update(trpo_args)
del cli_args['model']
