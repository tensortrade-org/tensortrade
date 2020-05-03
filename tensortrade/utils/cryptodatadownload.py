import pandas as pd
import ssl # Used if pandas gives a SSLError
ssl._create_default_https_context = ssl._create_unverified_context

import pprint
from datetime import datetime

class CryptoDataDownload:

    url = "https://www.cryptodatadownload.com/cdd/"
    
    # For trades/ticks, not candles
    tick_symbol_list = {
                'Binance': [
                           'BTC/USDT', 'ETH/USDT', 'LTC/USDT', 'LINK/USDT', 
                           'BNB/USDT', 'XRP/USDT', 'EOS/USDT', 'TRX/USDT', 
                           'NEO/USDT', 'ETC/USDT', 'XLM/USDT', 'BAT/USDT', 
                           'QTUM/USDT', 'ADA/USDT', 'XMR/USDT', 'ZEC/USDT', 
                           'DASH/USDT', 'BTT/USDT', 'MATIC/USDT', 'PAX/USDT', 
                           'CELR/USDT', 'ONE/USDT'
                           ],
                'Bitstamp': [
                            'BTC/USD','BTC/EUR',
                            'BCH/USD','BCH/EUR','BCH/BTC',
                            'ETH/USD','ETH/EUR','ETH/BTC',
                            'LTC/USD','LTC/EUR','LTC/BTC',
                            'XRP/USD','XRP/EUR','XRP/BTC'
                            ]
                }
    
    @classmethod
    def fetch_default(cls,
                      exchange_name, 
                      base_symbol, 
                      quote_symbol, 
                      timeframe, 
                      include_all_volumes=False):
        
        filename = "{}_{}{}_{}.csv".format(exchange_name, 
                                           base_symbol, 
                                           quote_symbol, 
                                           timeframe)
        base_vc = "Volume {}".format(base_symbol)
        new_base_vc = "volume_base"
        quote_vc = "Volume {}".format(quote_symbol)
        new_quote_vc = "volume_quote"

        df = pd.read_csv(cls.url + filename, skiprows=1)
        df = df[::-1]
        df = df.drop(["Symbol"], axis=1)
        df = df.rename({base_vc: new_base_vc, 
                        quote_vc: new_quote_vc, 
                        "Date": "date"}, axis=1)

        if "d" in timeframe:
            df["date"] = pd.to_datetime(df["date"])
        elif "h" in timeframe:
            df["date"] = pd.to_datetime(df["date"], 
                                        format="%Y-%m-%d %I-%p")

        df = df.set_index("date")
        df.columns = [name.lower() for name in df.columns]
        df = df.reset_index()
        if not include_all_volumes:
            df = df.drop([new_quote_vc], axis=1)
            df = df.rename({new_base_vc: "volume"}, axis=1)
            return df
        return df

    @classmethod
    def fetch_gemini(cls, base_symbol, quote_symbol, timeframe):
        exchange_name = "gemini"
        if timeframe.lower() in ['1d', 'd']:
            exchange_name = "Gemini"
            timeframe = 'd'
        elif timeframe.lower() == 'h':
            timeframe = timeframe[:-1] + "hr"

        filename = "{}_{}{}_{}.csv".format(exchange_name, 
                                           base_symbol, 
                                           quote_symbol, 
                                           timeframe)
        df = pd.read_csv(cls.url + filename, 
                         skiprows=1)
        df = df[::-1]
        df = df.drop(["Symbol", "Unix Timestamp"], axis=1)
        df.columns = [name.lower() for name in df.columns]
        df = df.set_index("date")
        df = df.reset_index()
        return df

    @classmethod
    def fetch_candles(cls,
                      exchange_name = 'Coinbase',
                      base_symbol = 'BTC',
                      quote_symbol = 'USD',
                      timeframe = '1d',
                      include_all_volumes = False): 
        """
            Fetch CSVs of Candle/OHLCV Data from CDD 
            Only 1d and 1h time frames are available
            There may be errors getting data from untested exchanges

            Check this link to see all the available exchanges:
                https://www.cryptodatadownload.com/data/

            Example Usage:
                from tensortrade.utils import CryptoDataDownload as cdd

            cdd.fetch_candles(exchange_name = 'Coinbase',
                              base_symbol = 'BTC',
                              quote_symbol = 'USD',
                              timeframe = '1h',
                              include_all_volumes = False)
        """
        if 'd' in timeframe.lower():
            timeframe = 'd'
            
        if exchange_name.lower() == "gemini":
            return cls.fetch_gemini(base_symbol,
                                    quote_symbol,
                                    timeframe)
        return cls.fetch_default(exchange_name, 
                                 base_symbol,
                                 quote_symbol, 
                                 timeframe, 
                                 include_all_volumes)
    @classmethod
    def fetch_trades(cls,
                     exchange = None, 
                     symbol = 'BTC/USDT', 
                     month = 'aug'):
        """
            Fetch CSVs of Tick/Trade Data from CDD 
            Quickly gets 300 000+ trades, for 36mb.

            Binance and Bitstamp are the only exchanges available.
            Check these links to see all the available pairs:
                https://www.cryptodatadownload.com/data/binance/
                https://www.cryptodatadownload.com/data/bitstamp/

            Example Usage:
                from tensortrade.utils.cryptodatadownload import CryptoDataDownload as cdd

                cdd.fetch_trades(exchange='binance', # or Bitstamp
                                 symbol='BTC/USDT', # run cdd.all() to see all
                                 month='aug') # Aug - Sep
        """

        # Parse date input
        months = ['August', 'September', 'October', 'November', 'December', 'January']
        month = month.strip(' ').lower() if month not in months else month
        for month_ in months:
            month_L = month_.lower()
            if month_L.startswith(month) or month_L.endswith(month) or month_L.find(month) >= 0:
                month = month_

        # Parse input symbol
        for delim in ['/', '-', '_', ' ']:
            try:
                base, quote = symbol.strip(' ').upper().split(delim)
                break
            except:
                continue

        if not base or not quote:
            print(f'Please input a symbol with tick data available')
            pprint.pprint(cls.symbol_list)
            return

        # Correct USD/T if Exchange is explicit 
        if exchange:
            exchange_ = exchange.lower()
            if ('binance'.startswith(exchange_) or 
                'binance'.endswith(exchange_) or 
                'binance'.find(exchange_) >= 0) and quote == 'USD':
                quote += 'T'
            elif ('bitstamp'.startswith(exchange_) or 
                  'bitstamp'.endswith(exchange_) or 
                  'bitstamp'.find(exchange_) >= 0) and quote == 'USDT':
                quote = 'USD'

        # Get proper exchange name
        for ex, ex_data in cls.tick_symbol_list.items():
            if base+'/'+quote in ex_data:
                exchange = ex
                break

        year = '2020' if month == 'January' else '2019' # Deduce Year

        exch_date = f'{month}{year}_{exchange}' if exchange == 'Binance' else f'{exchange}_{month}{year}'
        filename = f'tradeprints/{base}{quote}_{exch_date}_prints.csv'
        df = pd.read_csv(cls.url + filename, skiprows=1 if exchange == 'Binance' else 0)
        
        unix_multip = 1000 if exchange == 'Binance' else 1
        df['datetime'] = df['unix'].apply(lambda x: datetime.utcfromtimestamp(x/unix_multip))
        return df.set_index('datetime')
    
    @classmethod
    def all(cls): 
        """Print all available tick data symbols"""
        pprint.pprint(cls.tick_symbol_list)
