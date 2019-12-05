import talib
import sys
from talib.abstract import Function
# import numpy as np
import pandas as pd
from loguru import logger
config = {
    "handlers": [
        {"sink": sys.stdout},
    ]
}

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

    def _match_inputs(self, x_column:list, inputs:list):
        """ Search through inputs to match outputs. It only goes through common inputs """
        real_inputs = []
        for inp in inputs:
            if inp == "close":
                if inp in x_column:
                    real_inputs.append(inp)
                if "Close" in x_column:
                    real_inputs.append("Close")
            elif inp == "open":
                if inp in x_column:
                    real_inputs.append(inp)
                elif "Open" in x_column:
                    real_inputs.append("Open")
            elif inp == "high":
                if inp in x_column:
                    real_inputs.append(inp)
                elif "High" in x_column:
                    real_inputs.append("High")
            elif inp == "low":
                if inp in x_column:
                    real_inputs.append(inp)
                elif "Low" in x_column:
                    real_inputs.append("Low")
            elif inp == "volume":
                if inp in x_column:
                    real_inputs.append(inp)
                elif "VolumeTo" in x_column:
                    real_inputs.append("VolumeTo")
        return real_inputs

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for idx, indicator in enumerate(self._indicators):
            
            indicator_name = self._indicator_names[idx]
            logger.debug(indicator_name)
            indicator_params = self._stats[indicator_name]['parameters']
            indicator_inputs = self._stats[indicator_name]["inputs"]
            # Convert inputs into something that we'd commonly run to
            matched_inputs = self._match_inputs(list(X.columns), indicator_inputs)
            indicator_args = [X[arg].values for arg in matched_inputs]
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
