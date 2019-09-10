

from tensortrade.actions import *
from tensortrade.exchanges.simulated import *
from tensortrade.exchanges.live import *
from tensortrade.rewards import *
from tensortrade.slippage import *


def test_getting_components_by_name():

    checks_to_perform = [
        ('continuous', ContinuousActionStrategy),
        ('discrete', DiscreteActionStrategy),
        ('ccxt', CCXTExchange),
        ('simulated', SimulatedExchange),
        ('fbm', FBMExchange),
        ('gan', GANExchange),
        ('simple', SimpleProfitStrategy),
        ('uniform', RandomUniformSlippageModel)
    ]

    for registrar, name, clazz in checks_to_perform:
        assert isinstance(registrar.get(name), clazz)
