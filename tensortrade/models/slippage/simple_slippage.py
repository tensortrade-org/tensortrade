import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod
from typing import Tuple

from tensortrade.exchanges import AssetExchange
from tensortrade.models.slippage import SlippageModel


class SimpleSlippage(SlippageModel):
    def __init__(self):
        pass

    def fill_order(self, amount: float, price: float, exchange: AssetExchange) -> Tuple[float, float]:
        return amount, price
