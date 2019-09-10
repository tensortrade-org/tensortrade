

import tensortrade.actions as actions
import tensortrade.exchanges as exchanges
import tensortrade.rewards as rewards
import tensortrade.slippage as slippage


def test_getting_components_by_name():

    checks_to_perform = [
        (actions, 'continuous', 'ContinuousActionStrategy'),
        (actions, 'discrete', 'DiscreteActionStrategy'),
        (exchanges, 'ccxt', 'CCXTExchange'),
        (exchanges, 'simulated', 'SimulatedExchange'),
        (exchanges, 'fbm', 'FBMExchange'),
        (exchanges, 'gan', 'GANExchange'),
        (rewards, 'simple', 'SimpleProfitStrategy'),
        (slippage, 'uniform', 'RandomUniformSlippageModel')
    ]

    for registrar, name, class_name in checks_to_perform:
        assert registrar.get(name).__name__ == class_name
