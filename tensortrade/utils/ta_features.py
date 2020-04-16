# -*- coding: utf-8 -*-
"""ta_features.py

Interactive Colab Demo @
    https://colab.research.google.com/drive/1D86JximL2g9n6fTGmEkDhdip8Tw7T-9U

# 144 Unique Indicators
    Utilizing Tulip, Tulipy, PanTulipy and Pandas_TA (which includes the Bukosabino TA Indicators)
"""


## Requirements
# !pip install pandas-ta
# !pip install cython
# !pip install tulipy
# !pip install git+https://github.com/kodiakcrypto/pantulipy.git
# !pip install ccxt # for CryptoDataDownload

import pandas as pd

from .cryptodatadownload import CryptoDataDownload as cdd

# Access Tulip indicators w/default values
import pantulipy
from pantulipy import * 

import pandas_ta as ta


class TA_Features:
    """
        Get 144 Unique Indicator Values based on a DataFrame,
        with columns: 'open' 'high' 'low' 'close' 'volume'.
        Using Tulip indicators and Pandas_TA Indicators.
    """
    unique_pandas_ta_indicators = {
            'accbands': ta.accbands, 'amat': ta.amat, 'aobv': ta.aobv, 
            'cg': ta.cg, 'coppock': ta.coppock, 'decreasing': ta.decreasing, 
            'donchian': ta.donchian, 'efi': ta.efi, 'eom': ta.eom, 'fwma': ta.fwma, 
            'ichimoku': ta.ichimoku, 'increasing': ta.increasing, 'kc': ta.kc, 
            'kst': ta.kst, 'kurtosis': ta.kurtosis, 'linear_decay': ta.linear_decay, 
            'log_return': ta.log_return, 'mad': ta.mad, 'median': ta.median, 
            'midpoint': ta.midpoint, 'midprice': ta.midprice, 
            'percent_return': ta.percent_return, 'pvol': ta.pvol, 'pvt': ta.pvt, 
            'pwma': ta.pwma, 'quantile': ta.quantile, 'rma': ta.rma, 'rvi': ta.rvi, 
            'sinwma': ta.sinwma, 'skew': ta.skew, 'slope': ta.slope, 'swma': ta.swma, 
            't3': ta.t3, 'tsi': ta.tsi, 'uo': ta.uo, 'variance': ta.variance, 
            'vortex': ta.vortex, 'vp': ta.vp, 'vwap': ta.vwap, 'zscore': ta.zscore 
        }
    pandas_ta_indicators = {
            'accbands': ta.accbands, 'ad': ta.ad, 'adosc': ta.adosc, 'adx': ta.adx, 
            'amat': ta.amat, 'ao': ta.ao, 'aobv': ta.aobv, 'apo': ta.apo, 
            'aroon': ta.aroon, 'atr': ta.atr, 'bbands': ta.bbands, 'bop': ta.bop, 
            'cci': ta.cci, 'cg': ta.cg, 'cmo': ta.cmo, 
            'coppock': ta.coppock, 'decreasing': ta.decreasing, 'dema': ta.dema, 
            'donchian': ta.donchian, 'dpo': ta.dpo, 'efi': ta.efi, 'ema': ta.ema, 
            'eom': ta.eom, 'fisher': ta.fisher, 'fwma': ta.fwma, 'hma': ta.hma, 
            'ichimoku': ta.ichimoku, 'increasing': ta.increasing, 'kama': ta.kama, 
            'kc': ta.kc, 'kst': ta.kst, 'kurtosis': ta.kurtosis, 
            'linear_decay': ta.linear_decay, 'log_return': ta.log_return, 
            'macd': ta.macd, 'mad': ta.mad, 'massi': ta.massi, 'median': ta.median, 
            'mfi': ta.mfi, 'midpoint': ta.midpoint, 'midprice': ta.midprice, 
            'mom': ta.mom, 'natr': ta.natr, 'nvi': ta.nvi, 'obv': ta.obv, 
            'percent_return': ta.percent_return, 'ppo': ta.ppo, 'pvi': ta.pvi, 
            'pvol': ta.pvol, 'pvt': ta.pvt, 'pwma': ta.pwma, 'qstick': ta.qstick, 
            'quantile': ta.quantile, 'rma': ta.rma, 'roc': ta.roc, 'rsi': ta.rsi, 
            'rvi': ta.rvi, 'sinwma': ta.sinwma, 'skew': ta.skew, 'slope': ta.slope, 
            'sma': ta.sma, 'stdev': ta.stdev, 'stoch': ta.stoch, 'swma': ta.swma, 
            't3': ta.t3, 'tema': ta.tema, 'trima': ta.trima, 'trix': ta.trix, 
            'true_range': ta.true_range, 'tsi': ta.tsi, 'uo': ta.uo, 
            'variance': ta.variance, 'vortex': ta.vortex, 'vp': ta.vp, 
            'vwap': ta.vwap, 'vwma': ta.vwma, 'willr': ta.willr, 'wma': ta.wma, 
            'zlma': ta.zlma, 'zscore': ta.zscore 
        } 
    unique_pantulipy_indicators = {
            'adxr': adxr, 'aroonosc': aroonosc, 'avgprice': avgprice, 
            'cvi': cvi, 'decay': decay, 'di': di, 'dm': dm, 'dx': dx, 
            'edecay': edecay, 'emv': emv, 'fosc': fosc, 'kvo': kvo, 
            'lag': lag, 'linreg': linreg, 'linregintercept': linregintercept, 
            'linregslope': linregslope, 'marketfi': marketfi, 
            'md': md, 'msw': msw, 'psar': psar, 'rocr': rocr, 'stderr': stderr, 
            'tr': tr, 'tsf': tsf, 'typprice': typprice, 'vhf': vhf, 
            'vidya': vidya, 'volatility': volatility, 'vosc': vosc, 'wad': wad, 
            'wcprice': wcprice
        }
    pantulipy_indicators = {
            'ad': ad, 'adosc': adosc, 'adx': adx, 'adxr': adxr, 'ao': ao, 
            'apo': apo, 'aroon': aroon, 'aroonosc': aroonosc, 'atr': atr, 
            'avgprice': avgprice, 'bbands': bbands, 'bop': bop, 'cci': cci,
            'cmo': cmo, 'cvi': cvi, 'decay': decay, 'dema': dema, 'di': di, 
            'dm': dm, 'dpo': dpo, 'dx': dx, 'edecay': edecay, 'ema': ema, 
            'emv': emv, 'fisher': fisher, 'fosc': fosc, 'hma': hma, 
            'kama': kama, 'kvo': kvo, 'lag': lag, 'linreg': linreg, 
            'linregintercept': linregintercept, 'linregslope': linregslope, 
            'macd': macd, 'marketfi': marketfi, 'mass': mass, 'md': md, 
            'mfi': mfi, 'mom': mom, 'msw': msw, 'natr': natr, 'nvi': nvi, 
            'obv': obv, 'ppo': ppo, 'psar': psar, 'pvi': pvi, 'qstick': qstick, 
            'roc': roc, 'rocr': rocr, 'rsi': rsi, 'sma': sma, 'stderr': stderr, 
            'stoch': stoch, 'tema': tema, 'tr': tr, 'trima': trima, 'trix': trix, 
            'tsf': tsf, 'typprice': typprice, 'ultosc': ultosc, 'vhf': vhf, 
            'vidya': vidya, 'volatility': volatility, 'vosc': vosc, 'vwma': vwma, 
            'wad': wad, 'wcprice': wcprice, 'wilders': wilders, 'willr': willr, 
            'wma': wma, 'zlema': zlema 
        }

    ## Fetch Crypto Data Function
    def fetch(**kwargs):  
        """ 
            Wrapper Function for the CryptoDataDownload Class 
            Defaults and parameters:
                exchange_name: str = kwargs.get('exchange_name', 'Coinbase')
                base_symbol: str = kwargs.get('base_symbol', 'BTC')
                quote_symbol: str = kwargs.get('quote_symbol', 'USD')
                timeframe: str = kwargs.get('timeframe', '1h')
                include_all_volumes: bool = kwargs.get('include_all_volumes', False)
        """ 
        df = cdd.fetch(**kwargs)

        return df

    def _remove_duplicate_columns(df: pd.DataFrame):
        """Rename all duplicate columns appending _2 or _3 or _4 etc"""
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique(): 
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i+1) if i != 0 else dup for i in range(sum(cols == dup))]
        df.columns = cols # Rename columns
        return df
    
    def _append_column(orig, new):
        if isinstance(new, pd.DataFrame):
            for n, column in enumerate(new.columns):
                orig[column.lower()] = new.iloc[:,n]
        else:
            orig[new.name.lower()] = new
        return orig

    def get_all_pandas_ta_indicators(data=None, unique=False, **kwargs):
        """
            Get all Pandas_TA indicators
            :param: data - DataFrame with columns: 'open' 'high' 'low' 'close' 'volume'
            :param: unique - True would exclude all indicators that Tulip also has

            kwargs and defaults for CryptoDataDownload Class:
                exchange_name: str = 'Coinbase'
                base_symbol: str = 'BTC'
                quote_symbol: str = 'USD'
                timeframe: str = '1h'
                include_all_volumes: bool = False
        """

        data = TA_Features.fetch(**kwargs) if not type(data) == pd.DataFrame or data.empty else data

        open_ = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        ind = TA_Features.unique_pandas_ta_indicators if unique else TA_Features.pandas_ta_indicators

        for name, function in ind.items():
            if name == 'massi': # Can't handle being sent a close= kwarg
                indicator_data = function(high=high,
                                          low=low)
            else:
                indicator_data = function(open_=open_,
                                          high=high,
                                          low=low,
                                          close=close,
                                          volume=volume)
            if type(indicator_data) == tuple: # tuple of dataframes
                for i in range(0, len(indicator_data)):
                    data = TA_Features._append_column(data, indicator_data[i])
            else:
                data = TA_Features._append_column(data, indicator_data)
        return TA_Features._remove_duplicate_columns(data)

    def get_all_pantulipy_indicators(data=None, unique=False, **kwargs):
        """
            Get all Tulip Indicators
            :param: data - DataFrame with columns: 'open' 'high' 'low' 'close' 'volume'
            :param: unique - True would exclude all indicators that Pandas_TA has also 

            kwargs and defaults for CryptoDataDownload Class:
                exchange_name: str = 'Coinbase'
                base_symbol: str = 'BTC'
                quote_symbol: str = 'USD'
                timeframe: str = '1h'
                include_all_volumes: bool = False
        """
        data = TA_Features.fetch(**kwargs) if not type(data) == pd.DataFrame or data.empty else data

        ind = TA_Features.unique_pantulipy_indicators if unique else TA_Features.pantulipy_indicators
        for name, function in ind.items():
            if name not in pantulipy.core._DEFAULTLESS_INDICATORS:
                data = pd.concat([data, function(data)], axis=1)
        return TA_Features._remove_duplicate_columns(data)

    def get_all_indicators(data=None, **kwargs):
        """
            Get all Tulipy and Pandas_TA indicators
            :param: data = DataFrame with columns: 'open' 'high' 'low' 'close' 'volume'

            kwargs and defaults for CryptoDataDownload Class:
                exchange_name: str = 'Coinbase'
                base_symbol: str = 'BTC'
                quote_symbol: str = 'USD'
                timeframe: str = '1h'
                include_all_volumes: bool = False
        """
        data = TA_Features.fetch(**kwargs) if not type(data) == pd.DataFrame or data.empty else data

        data = TA_Features.get_all_pandas_ta_indicators(data, unique=True)
        data = TA_Features.get_all_pantulipy_indicators(data, unique=False)
        return data
