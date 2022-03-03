#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File:        tensortrade/binance/pair_at_interval.py
# By:          Samuel Duclos
# For:         Myself
# Description: This file is the top-level class for manipulating a selected 
#              cryptocurrency pair within a given interval and period through 
#              its instanciated object.

#from .triggers import Binance_triggers

import datetime
import pandas as pd

class Binance_pair_at_interval:
    def __init__(self, spot_client, futures_client, pair, interval):
        self.spot_client = spot_client
        self.dataset = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        self.interval = interval
        self.period = self.get_n_periods_from_time(n=60)
        #self.indicators = Binance_triggers(futures_client=futures_client)

    def update(self, pair, download=True):
        if download:
            self.dataset = self.download_dataset(pair=pair)

            #try:
            #    self.indicators.update(self.dataset, pair)
            #except IndexError:
            #    pass

            #self.indicators.update(self.dataset, pair)

    def get_n_periods_from_time(self, n=60):
        return str(int(self.interval[:-1]) * n) + self.interval[-1:]

    def download_dataset(self, pair):
        dataset = \
            self.spot_client.get_historical_klines(symbol=pair, 
                                                   interval=self.interval, 
                                                   start_str=self.period)

        dataset = pd.DataFrame(dataset, 
                               columns=['time', 
                                        'open', 
                                        'high', 
                                        'low', 
                                        'close', 
                                        'volume', 
                                        'Close time', 
                                        'Quote asset volume', 
                                        'Number of trades', 
                                        'Taker buy base asset volume', 
                                        'Taker buy quote asset volume', 
                                        'Ignore'])

        four_hours = 14400
        milliseconds = 1000

        dataset['time'] = \
            dataset['time'].apply(lambda timestamp: \
                                  datetime.datetime.fromtimestamp((timestamp / \
                                                                   milliseconds) - \
                                                                  four_hours))

        dataset = dataset[['time', 'open', 'high', 'low', 'close', 'volume']]
        dataset.set_index('time', inplace=True)
        return dataset.applymap(lambda entry: entry.rstrip('0').rstrip('.'))

