import xagents
from xagents.base import OffPolicy
from xagents.utils.cli import agent_args, non_agent_args, off_policy_args


def get_expected_flags(argv, as_kwargs=False):
    """
    Convert argv to expected flags/keyword that should be present
    as argparse.Namespace attributes and as kwargs that will be
    passed directly to agent.
    Args:
        argv: Arguments passed.
        as_kwargs: If True example-flag1 will be returned as example_flag1

    Returns:
        List of ['example-flag1', 'example-flag2', ...]
        or
        list of ['example_flag1', 'example_flag2']
    """
    if not argv:
        return []
    command = argv[0]
    expected_kwargs = {}
    expected_kwargs.update(agent_args)
    expected_kwargs.update(non_agent_args)
    expected_kwargs.update(xagents.commands[command][0])
    if len(argv) > 1:
        agent_data = xagents.agents[argv[1]]
        expected_kwargs.update(agent_data['module'].cli_args)
        if issubclass(agent_data['agent'], OffPolicy) or argv[1] == 'acer':
            expected_kwargs.update(off_policy_args)
    if not as_kwargs:
        return expected_kwargs.keys()
    return [flag.replace('-', '_') for flag in expected_kwargs.keys()]


def assert_flags_displayed(cap, title, cli_args):
    """
    Assert title and respective flags are present in help menu.
    Args:
        cap: str, text displayed to the console
        title: str, that should be present in cap representing
            the title of the group of argument group.
        cli_args: A dictionary of flags and their attributes.

    Returns:
        None
    """
    assert title in cap
    for flag in cli_args:
        assert f'--{flag}' in cap
        for key in cli_args[flag]:
            if key in ['help', 'hp_type', 'default']:
                for line in str(cli_args[flag][key]).split('\n'):
                    assert line in cap
