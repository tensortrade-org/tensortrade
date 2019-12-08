

import importlib

from .simple_moving_average import SimpleMovingAverage

if importlib.util.find_spec('ta') is not None:
    from .ta_indicator import TAIndicator

if importlib.util.find_spec("talib") is not None:
    from .talib_indicator import TAlibIndicator
