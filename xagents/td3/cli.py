from xagents import ddpg

ddpg_args = ddpg.cli_args
td3_args = {
    'policy-delay': {
        'help': 'Delay after which, actor weights and target models will be updated',
        'type': int,
        'default': 2,
        'hp_type': 'categorical',
    },
    'policy-noise-coef': {
        'help': 'Coefficient multiplied by noise added to target actions',
        'type': float,
        'default': 0.2,
        'hp_type': 'log_uniform',
    },
    'noise-clip': {
        'help': 'Target noise clipping value',
        'type': float,
        'default': 0.5,
        'hp_type': 'log_uniform',
    },
}
cli_args = ddpg_args.copy()
cli_args.update(td3_args)
