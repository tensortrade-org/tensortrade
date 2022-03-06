#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File:        tensortrade/binance/strategy.py
# By:          Samuel Duclos
# For:         Myself
# Description: This file handles the main strategy calculations.

import pandas as pd

class Binance_strategy:
    def __init__(self, crypto):
        self.crypto = crypto

    def filter_pairs_connected_to_held_asset(self):
        self.crypto.pair.info.calculate_balance()
        self.crypto.pair.calculate_side()

        if self.crypto.pair.side == 'BUY':
            base_eq_base = \
                self.crypto.info_all.exchange_info[
                    self.crypto.info_all.exchange_info['baseAsset'] == self.crypto.pair.info.base_asset
                ]
            quote_eq_base = \
                self.crypto.info_all.exchange_info[
                    self.crypto.info_all.exchange_info['quoteAsset'] == self.crypto.pair.info.base_asset
                ]
            related_pairs = pd.concat([base_eq_base, quote_eq_base], axis='index').reset_index(drop=True)

        elif self.crypto.pair.side == 'SELL':
            base_eq_quote = \
                self.crypto.info_all.exchange_info[
                    self.crypto.info_all.exchange_info['baseAsset'] == self.crypto.pair.info.quote_asset
                ]
            quote_eq_quote = \
                self.crypto.info_all.exchange_info[
                    self.crypto.info_all.exchange_info['quoteAsset'] == self.crypto.pair.info.quote_asset
                ]
            related_pairs = pd.concat([base_eq_quote, quote_eq_quote], axis='index').reset_index(drop=True)

        return related_pairs

    def get_moving_pairs(self, 
                         tickers=None, 
                         pair=None, 
                         count=5, 
                         ascending=False, 
                         threshold_extrema=None, 
                         volume_threshold_extrema=None, 
                         shuffle=False, 
                         short=False):

        def get_new_tickers():
            tickers = pd.DataFrame(self.crypto.spot_client.get_ticker())
            tickers[['priceChangePercent', 'volume', 'bidPrice', 'askPrice', 'bidQty', 'askQty']] = \
                tickers[['priceChangePercent', 'volume', 'bidPrice', 'askPrice', 'bidQty', 'askQty']].astype(float)
            return tickers

        def get_tickers_long(pair=None):
            related_pairs = self.filter_pairs_connected_to_held_asset()
            related_pairs = related_pairs['symbol']
            tickers = get_new_tickers()
            tickers = tickers[tickers['symbol'].isin(related_pairs)]
            tickers = tickers.reset_index(drop=True)
            tickers = tickers[tickers['volume'] > 500]
            tickers['bidAskChangePercent'] = (tickers['askPrice'] - tickers['bidPrice']) / tickers['askPrice']
            tickers['bidAskQtyChangePercent'] = (tickers['askQty'] - tickers['bidQty']) / tickers['bidQty']
            tickers['bidAskChangePercent'] = tickers['bidAskChangePercent'].apply(abs)
            tickers[['bidAskChangePercent', 'bidAskQtyChangePercent']] *= 100
            tickers = tickers.fillna(0)
            tickers = tickers[tickers['bidAskChangePercent'] < 0.1]
            tickers = tickers[tickers['bidAskQtyChangePercent'] > 0.0]
            tickers['volumeMarketPercent'] = tickers['volume'] / tickers['volume'].sum()
            tickers['experimental'] = tickers['bidAskQtyChangePercent'] * tickers['volumeMarketPercent']
            tickers = tickers.sort_values(by='experimental', ascending=False)
            if pair is not None:
                quote_asset_pair_info = pair.info.exchange_info[pair.info.exchange_info['quoteAsset'] == pair.info.quote_asset]
                tickers = tickers[tickers['symbol'].isin(quote_asset_info_pairs['symbol'])]
                tickers = tickers[['symbol', 'priceChangePercent', 'lastPrice', 'volume', 'bidAskQtyChangePercent']]
                quote_asset_pair_info = quote_asset_pair_info.reset_index(drop=True)
                tickers = tickers.reset_index(drop=True)
                return tickers, quote_asset_pair_info
            else:
                tickers = tickers[['symbol', 
                                   'priceChangePercent', 
                                   'lastPrice', 
                                   'volume', 
                                   'bidAskQtyChangePercent', 
                                   'count']]
                tickers = tickers.reset_index(drop=True)
                return tickers, None

        def get_tickers_short(pair=None):
            tickers = pd.DataFrame(crypto.spot_client.get_ticker())
            tickers[['priceChangePercent', 'bidQty', 'askQty']] = \
                tickers[['priceChangePercent', 'bidQty', 'askQty']].astype(float)
            tickers = tickers[tickers['bidQty'] > tickers['askQty']]
            tickers = tickers[['symbol', 'priceChangePercent']]
            return tickers[tickers['symbol'].str.endswith(quote_asset)], None

        get_tickers = get_tickers_short if short else get_tickers_long

        if tickers is None:
            tickers, quote_asset_pair_info = get_tickers()
            return tickers, quote_asset_pair_info, None

        else:
            old_tickers = tickers.copy()
            tickers, quote_asset_pair_info = get_tickers()

            tickers['price_difference'] = tickers['priceChangePercent'] - old_tickers['priceChangePercent']
            tickers = tickers.sort_values(by=['price_difference'], ascending=False)
            tickers = tickers.reset_index(drop=True)

            moving_pairs = tickers['symbol']

            if threshold_extrema is not None:
                moving_pairs = moving_pairs[threshold_extrema(tickers['price_difference'])]

            moving_pairs = moving_pairs.iloc[:count]

            if shuffle:
                moving_pairs = moving_pairs.sample(frac=1.0)

            moving_pairs = moving_pairs.tolist()

            return tickers, quote_asset_pair_info, moving_pairs

