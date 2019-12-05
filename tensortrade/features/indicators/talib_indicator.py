import talib
from talib.abstract import Function
import numpy as np
import pandas as pd

from gym import Space
from copy import copy
from abc import abstractmethod
from typing import Union, List, Callable, Dict

from tensortrade.features import FeatureTransformer


class TAlibIndicator(FeatureTransformer):
    """Adds one or more TAlib indicators to a data frame, based on existing open, high, low, and close column values."""

    def __init__(self, indicators: List[str], lows: Union[List[float], List[int]] = None, highs: Union[List[float], List[int]] = None, **kwargs):
        indicators = self._error_check(indicators)
        self._indicator_names = [indicator.upper() for indicator in indicators]
        self._indicators = [getattr(talib, name.split('-')[0]) for name in self._indicator_names]
        # Here we get the stats for each indicator for TA-Lib
        self._stats = {indicator:self._get_info(indicator) for indicator in self._indicator_names}
        
    def _error_check(self, a:List[str])->List[str]:
        """ Check for errors common errors"""
        err_indexes = []
        for n, i in enumerate(a):
            if i == "BBAND":
                a[n] = "BBANDS"
            elif i == "BB":
                a[n] = "BBANDS"
            elif i == "RIS":
                a[n] = "RSI"
            elif i == "":
                err_indexes.append(n)
            elif i == None:
                err_indexes.append(n)
        for n in sorted(err_indexes, reverse=True):
            del a[n]
        return a
    

    def _get_info(self, indicator_name:str) -> Dict:
        """ Get the relavent indicator parameters and inputs """
        if indicator_name is None:
            print("Usage: help_indicator(symbol), symbol is indicator name")
            return {
                "parameters": {},
                "inputs": []
            }
        else:
            upper_code = indicator_name.upper()
            if upper_code not in talib.get_functions():
                print(f"ERROR: indicator {upper_code} not in list")
                return {
                    "parameters": {},
                    "inputs": []
                }
            else:
                func = Function(upper_code)
                parameters = dict(func.parameters)
                inputs = list(func.input_names.values())
                return {
                    "parameters": parameters,
                    "inputs": inputs
                }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for idx, indicator in enumerate(self._indicators):
            indicator_name = self._indicator_names[idx]
            indicator_params = self._stats[indicator_name]['parameters']
            indicator_args = [X[arg].values for arg in self._stats[indicator_name]["inputs"]]
            
            if indicator_name == 'BBANDS':
                upper, middle, lower = indicator(*indicator_args,**indicator_params)

                X["bb_upper"] = upper
                X["bb_middle"] = middle
                X["bb_lower"] = lower
            else:
                try:
                    value = indicator(*indicator_args,**indicator_params)

                    if type(value) == tuple:
                        X[indicator_name] = value[0][0]
                    else:
                        X[indicator_name] = value

                except:
                    X[indicator_name] = indicator(*indicator_args,**indicator_params)[0]

        return X
