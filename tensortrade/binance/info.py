#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File:        tensortrade/binance/info.py
# By:          Samuel Duclos
# For:         Myself
# Description: This file provides top-level access to the whole Binance 
#              crypto exchange info.

import pandas as pd

class Binance_info:
    """
    exchange_info_object = Binance_info(spot_client=spot_client)
    exchange_info = exchange_info_object.exchange_info
    tradable_tickers = exchange_info_object.get_tradable_tickers_info()
    print(exchange_info)
    print(tradable_tickers)
    """
    def __init__(self, spot_client):
        self.spot_client = spot_client
        self.exchange_info = self.get_exchange_info()

    def get_exchange_info(self):
        symbols_info = self.spot_client.get_exchange_info()
        symbols_info = pd.DataFrame([symbols_info])
        symbols_info = symbols_info['symbols']
        symbols_info = symbols_info.iloc[0]
        symbols_info = pd.DataFrame(symbols_info)
        symbols_info = symbols_info[symbols_info['status'] == 'TRADING']
        symbols_info = symbols_info[symbols_info['isSpotTradingAllowed']]
        symbols_info = symbols_info[symbols_info['ocoAllowed']]
        symbols_info = symbols_info[symbols_info['quoteOrderQtyMarketAllowed']]
        symbols_info = symbols_info.drop(columns=['status', 
                                                  'isSpotTradingAllowed', 
                                                  'isMarginTradingAllowed', 
                                                  'permissions', 
                                                  'icebergAllowed', 
                                                  'ocoAllowed', 
                                                  'quoteOrderQtyMarketAllowed', 
                                                  'orderTypes'])
    
        symbols_info = symbols_info.set_index('symbol', drop=False)
    
        df = pd.DataFrame()
        for i in range(symbols_info.shape[0]):
            df = pd.concat([df, self.build_filters(symbols_info, i)], axis='index')
        df = df.set_index('symbol')
        symbols_info = pd.concat([symbols_info, df], axis='columns')
        symbols_info = symbols_info.drop(columns=['symbol', 'filters'])
        symbols_info = symbols_info.reset_index('symbol')
        return symbols_info

    def build_filters(self, symbols_info, index):
        symbol = symbols_info['symbol'].iloc[index]
        df = pd.DataFrame(symbols_info['filters'].iloc[index])
        min_price = df[df['filterType'] == 'PRICE_FILTER']['minPrice'].iloc[0]
        max_price = df[df['filterType'] == 'PRICE_FILTER']['maxPrice'].iloc[0]
        tick_size = df[df['filterType'] == 'PRICE_FILTER']['tickSize'].iloc[0]
        step_size = df[df['filterType'] == 'LOT_SIZE']['stepSize'].iloc[0]
        multiplier_up = df[df['filterType'] == 'PERCENT_PRICE']['multiplierUp'].iloc[0]
        multiplier_down = df[df['filterType'] == 'PERCENT_PRICE']['multiplierDown'].iloc[0]
        df = pd.DataFrame([[symbol, min_price, max_price, tick_size, step_size, multiplier_up, multiplier_down]], 
                          columns=['symbol', 'min_price', 'max_price', 'tick_size', 'step_size', 'multiplier_up', 'multiplier_down'])
        return df

    def get_tradable_tickers_info(self):
        tickers = pd.DataFrame(self.spot_client.get_ticker())
        tickers[['priceChangePercent', 'lastPrice', 'volume', 'bidPrice', 'askPrice', 'bidQty', 'askQty', 'count']] = \
            tickers[['priceChangePercent', 'lastPrice', 'volume', 'bidPrice', 'askPrice', 'bidQty', 'askQty', 'count']].astype(float)
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
        tickers = tickers[['symbol', 'priceChangePercent', 'lastPrice', 'volume', 'bidAskQtyChangePercent', 'count']]
        tickers = tickers.reset_index(drop=True)
        return tickers

    def get_new_tickers(self):
        tickers = pd.DataFrame(self.spot_client.get_ticker())
        tickers[['priceChangePercent', 'volume', 'bidPrice', 'askPrice', 'bidQty', 'askQty']] = \
            tickers[['priceChangePercent', 'volume', 'bidPrice', 'askPrice', 'bidQty', 'askQty']].astype(float)
        return tickers

    def filter_pairs_connected_to_held_asset(self):
        crypto.pair.info.calculate_balance()
        crypto.pair.calculate_side()

        if crypto.pair.side == 'BUY':
            base_eq_base = crypto.info_all.exchange_info[crypto.info_all.exchange_info['baseAsset'] == crypto.pair.info.base_asset]
            quote_eq_base = crypto.info_all.exchange_info[crypto.info_all.exchange_info['quoteAsset'] == crypto.pair.info.base_asset]
            related_pairs = pd.concat([base_eq_base, quote_eq_base], axis='index').reset_index(drop=True)

        elif crypto.pair.side == 'SELL':
            base_eq_quote = crypto.info_all.exchange_info[crypto.info_all.exchange_info['baseAsset'] == crypto.pair.info.quote_asset]
            quote_eq_quote = crypto.info_all.exchange_info[crypto.info_all.exchange_info['quoteAsset'] == crypto.pair.info.quote_asset]
            related_pairs = pd.concat([base_eq_quote, quote_eq_quote], axis='index').reset_index(drop=True)

        return related_pairs

