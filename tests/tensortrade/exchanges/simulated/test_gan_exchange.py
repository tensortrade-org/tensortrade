from typing import Generator, List, Dict

import pandas as pd
from gym import Space
import pytest

from tensortrade import TradingContext
from tensortrade.exchanges import Exchange
from tensortrade.trades import Trade



@pytest.mark.xskip(reason="GAN exchange is not complete. ")
def test_create_gan_exchnage():
    """ GAN is not complete. Will not do this. """
    pass