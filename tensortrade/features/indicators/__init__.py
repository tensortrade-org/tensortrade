

import importlib

if importlib.util.find_spec("talib") is not None:
    from .simple_moving_average import SimpleMovingAverage
    from .talib_indicator import TAlibIndicator
