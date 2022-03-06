#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File:        tensortrade/binance/pair_info.py
# By:          Samuel Duclos
# For:         Myself
# Description: This file gets a selected cryptocurrency pair's info.

from math import floor
from pandas import DataFrame

class Binance_pair_info:
    def __init__(self, info, pair):
        self.spot_client = info.spot_client
        self.pair = pair
        self.info_all = info
        self.exchange_info = self.info_all.exchange_info

        pair_exchange_info = self.exchange_info[self.exchange_info['symbol'] == self.pair]
        self.base_asset = pair_exchange_info['baseAsset'].iloc[0]
        self.quote_asset = pair_exchange_info['quoteAsset'].iloc[0]
        self.precision = pair_exchange_info['quotePrecision'].iloc[0]
        self.base_asset_precision = pair_exchange_info['quoteAssetPrecision'].iloc[0]

        self.tick_size = float(pair_exchange_info['tick_size'])
        self.step_size = float(pair_exchange_info['step_size'])

    def compact_float_string(self, number, precision):
        return "{:0.0{}f}".format(number, precision).rstrip('0').rstrip('.')

    def make_tradable_quantity(self, coins_available):
        quantity = floor(coins_available * 10**self.step_size) / float(10**self.step_size)

        if self.tick_size < 0:
            quantity = floor(coins_available * abs(self.tick_size)) / float(abs(self.tick_size))

        return self.compact_float_string(float(quantity), self.precision)

    def calculate_balance(self):
        balances = DataFrame(self.spot_client.get_account()['balances'])[['asset', 'free']].set_index('asset').astype(float).T
        self.base_asset_balance = balances[self.base_asset].iloc[0]
        self.quote_asset_balance = balances[self.quote_asset].iloc[0]

        self.pair_last_price = float(self.spot_client.get_ticker(symbol=self.pair)['lastPrice'])
        self.pair_buy_balance = self.quote_asset_balance / self.pair_last_price
        self.pair_sell_balance = self.base_asset_balance * self.pair_last_price
        self.pair_combined_base_balance = self.pair_buy_balance + self.base_asset_balance
        self.pair_combined_quote_balance = self.pair_sell_balance + self.quote_asset_balance

        self.base_asset_balance = self.compact_float_string(self.base_asset_balance, self.base_asset_precision)
        self.quote_asset_balance = self.compact_float_string(self.quote_asset_balance, self.precision)
        self.pair_last_price = self.compact_float_string(self.pair_last_price, self.precision)
        self.pair_buy_balance = self.compact_float_string(self.pair_buy_balance, self.base_asset_precision)
        self.pair_sell_balance = self.compact_float_string(self.pair_sell_balance, self.precision)
        self.pair_combined_base_balance = self.compact_float_string(self.pair_combined_base_balance, self.base_asset_precision)
        self.pair_combined_quote_balance = self.compact_float_string(self.pair_combined_quote_balance, self.precision)

    def print_balance(self):
        print("\n", 'pair: ', self.pair)
        print('base_asset_balance: ', self.base_asset_balance)
        print('quote_asset_balance: ', self.quote_asset_balance)
        print('pair_last_price: ', self.pair_last_price)
        print('pair_buy_balance: ', self.pair_buy_balance)
        print('pair_sell_balance: ', self.pair_sell_balance)
        print('pair_combined_base_balance: ', self.pair_combined_base_balance)
        print('pair_combined_quote_balance: ', self.pair_combined_quote_balance, "\n")

