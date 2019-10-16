from . import actions
from . import environments
from . import exchanges
from . import features
from . import rewards
from . import slippage
from . import strategies
from . import trades

from .version import __version__
from gym.envs.registration import register

register(
    id='tensortrade-v0',
    entry_point='tensortrade.environments:TradingEnvironment',
)
