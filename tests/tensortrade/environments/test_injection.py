
from tensortrade import TradingContext
from tensortrade.environments import TradingEnvironment


config = {
    'base_instrument': 50,
    'instruments': ['BTC', 'ETH'],
    'actions': {
        'n_actions': 50,
        'max_allowed_slippage': 2.0
    },
    'exchanges': {

    }
}


def make_env(exchange: str, action: str, reward: str):
    return TradingEnvironment(exchange=exchange, action_scheme=action, reward_scheme=reward)


def test_injects_simulated_discrete_simple_environment():
    env = make_env('simulated', 'discrete', 'simple')

    assert env.action_scheme.n_actions == 20

    with TradingContext(**config):
        env = make_env('simulated', 'discrete', 'simple')
        assert env.action_scheme.n_actions == 50
