#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File:        tensortrade/binance/binance.py
# By:          Samuel Duclos
# For:         Myself
# Description: This file is the class for instanciating top-level objects 
#              giving access to the whole crypto trading system on Binance.

from .authentication import Binance_authenticator
from .info import Binance_info
from .pair import Binance_pair

class Binance:
    def __init__(self, spot_client=None, use_futures=False):
        keys_path = 'keys.txt'
        client = Binance_authenticator(spot_client=spot_client, 
                                              futures_client=None, 
                                              get_spot_client=True, 
                                              get_futures_client=use_futures, 
                                              keys_path=keys_path)

        self.spot_client = client.spot_client
        self.futures_client = client.futures_client
        self.info_all = Binance_info(spot_client=self.spot_client)

    def make_pair(self, symbol, intervals=['5m', '30m'], download=True):
        self.pair = Binance_pair(spot_client=self.spot_client, 
                                 futures_client=self.futures_client, 
                                 info_all=self.info_all, 
                                 pair=symbol, 
                                 intervals=intervals, 
                                 download=download)

