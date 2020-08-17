import importlib

if importlib.util.find_spec("stochastic") is not None:
    
    from .utils import *

    from .processes.cox import cox
    from .processes.fbm import fbm
    from .processes.gbm import gbm
    from .processes.heston import heston
    from .processes.merton import merton
    from .processes.ornstein_uhlenbeck import ornstein
