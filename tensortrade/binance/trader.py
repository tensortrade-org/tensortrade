#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File:        tensortrade/binance/trader.py
# By:          Samuel Duclos
# For:         Myself
# Description: This file handles raw trading (buying and selling) 
#              for a given pair.

from .pair_info import Binance_pair_info

class Binance_trader:
    def __init__(self, spot_client, info_all, pair):
        self.spot_client = spot_client
        self.side = 'SELL'
        self.default_strategy = 'LONG'
        self.strategy = 'HOLD'
        self.info = Binance_pair_info(info=info_all, pair=pair)

    def calculate_side(self):
        if float(self.info.pair_buy_balance) > float(self.info.base_asset_balance):
            self.side = 'SELL'
        elif float(self.info.base_asset_balance) > float(self.info.pair_buy_balance):
            self.side = 'BUY'

    def trade_pair(self, percentage_to_trade=1.0):
        self.info.calculate_balance()
        self.calculate_side()

        if self.side == 'SELL':
            coins_available = float(self.info.pair_buy_balance)
            side = 'BUY'
        elif self.side == 'BUY':
            coins_available = float(self.info.base_asset_balance)
            side = 'SELL'

        quantity = self.info.make_tradable_quantity(coins_available * percentage_to_trade)

        print('traded quantity:', quantity)
        self.info.calculate_balance()
        print('\nSide for base asset ' + self.info.base_asset + ' is ' + self.side + '.')
        print('Side for quote asset ' + self.info.quote_asset + ' is ' + side + '.')
        self.spot_client.create_order(symbol=self.info.pair, side=side, type='MARKET', quantity=quantity, recvWindow=2500)

        if self.strategy == 'HOLD':
            self.strategy = self.default_strategy
        if self.strategy == 'LONG':
            self.strategy = 'HOLD'

        print('\nSide for base asset ' + self.info.base_asset + ' is ' + side + '.')
        print('Side for quote asset ' + self.info.quote_asset + ' is ' + self.side + '.')
        self.side = side

