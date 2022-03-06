#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File:        tensortrade/binance/pair.py
# By:          Samuel Duclos
# For:         Myself
# Description: This file is the top-level class for manipulating a selected 
#              cryptocurrency pair through its instanciated object.

from .pair_at_interval import Binance_pair_at_interval
from .trader import Binance_trader

class Binance_pair(Binance_trader):
    def __init__(self, 
                 spot_client, 
                 futures_client, 
                 info_all, 
                 pair, 
                 intervals=['5m', '30m'], 
                 download=True):

        self.spot_client = spot_client
        self.futures_client = futures_client
        super().__init__(spot_client=self.spot_client, info_all=info_all, pair=pair)
        self.interval = self.get_datasets(intervals=intervals)

    def update(self, download=True):
        if download:
            self.info.calculate_balance()
            self.calculate_side()
            for interval in list(self.interval.keys()):
                self.interval[interval].update(pair=self.info.pair, download=download)

    def get_datasets(self, intervals=['1m'], download=False):
        dataset = dict()
        for interval in intervals:
            dataset[interval] = Binance_pair_at_interval(spot_client=self.spot_client, 
                                                         futures_client=self.futures_client, 
                                                         pair=self.info.pair, 
                                                         interval=interval)
        return dataset

