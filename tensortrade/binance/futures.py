#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File:        tensortrade/binance/futures.py
# By:          Samuel Duclos
# For:         Myself
# Description: This file handles futures-based calculations.

from contextlib import contextmanager
from binance_f.constant.test import *

import time
import sys
import os
import pandas as pd

class Binance_futures:
    def __init__(self, futures_client):
        self.futures_client = futures_client

    @contextmanager
    def suppress_stdout(self):
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

    def filter_buy_sell_ratio(self, symbol='BTCUSDT', period='1d'):
        time.sleep(0.05)
        with self.suppress_stdout():
            result = self.futures_client.get_taker_buy_sell_ratio(symbol=symbol, period=period)
        df = pd.DataFrame([[res.timestamp, res.buySellRatio] for res in result], columns=['time', 'buy_sell_ratio'])
        df = df.set_index('time').sort_index().reset_index(drop=True)
        return df.shape[0] != 0 and \
            (df['buy_sell_ratio'].iloc[-1] > 1.0 and \
             df['buy_sell_ratio'].iloc[-1] > df['buy_sell_ratio'].iloc[-2])

    def filter_top_long_short_accounts(self, symbol='BTCUSDT', period='1d'):
        time.sleep(0.05)
        with self.suppress_stdout():
            results = self.futures_client.get_top_long_short_accounts(symbol=symbol, period=period)
        df = pd.DataFrame([result.__dict__ for result in results])[['timestamp', 'longShortRatio']]
        df = df.rename(columns={'timestamp': 'time', 'longShortRatio': 'long_short_ratio'})
        df = df.set_index('time').sort_index().reset_index(drop=True)
        return df.shape[0] != 0 and \
            (df['long_short_ratio'].iloc[-1] > 1.0 and \
             df['long_short_ratio'].iloc[-1] > df['long_short_ratio'].iloc[-2])

    def get_futures_trigger(self, symbol='BTCUSDT', period='1d'):
        top_long_short_accounts_trigger = self.filter_top_long_short_accounts(symbol=symbol, period=period)
        buy_sell_ratio = self.filter_buy_sell_ratio(symbol=symbol, period=period)
        return top_long_short_accounts_trigger and buy_sell_ratio

