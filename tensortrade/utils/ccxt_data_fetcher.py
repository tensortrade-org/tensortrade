# -*- coding: utf-8 -*-
"""ccxt_data_fetcher.ipynb

Use this URL for a Google Colab Preview of this class and its usage:
    https://colab.research.google.com/drive/154_2tvDn_36pZzU_XkSv9Xvd3KjQCw1U
"""

import ccxt

from datetime import timedelta, datetime, timezone
import sys, os, time, random
import pandas as pd
import json

import csv
import sqlite3
from sqlite3 import Error

class CCXT_Data_Fetcher():
    """
        Majority Code Credit goes to: 
            https://github.com/Celeborn2BeAlive/cryptobigbro
            
        exchange_id: Any exchange id available thru CCXT 
                     https://github.com/ccxt/ccxt/wiki/Manual#exchanges

        symbol: A slash is used for all symbols except on BitMEX Futures
                eg. XRPH20 has no slash
                but XBT/USD and ETH/USD are how they identify the USD pairs.

        timeframe: Any timeframe available on the chosen exchange.
        candle_amount: Use 'all' to get FULL candle history. Default is 500.
        
        save_path: Use if you want to save data as a .csv file or SQLite DB.
        save_format: 'csv' or 'sqlite' are the database options.
        
        Example Usage:
            from tensortrade.utils.ccxt_data_fetcher import CCXT_Data_Fetcher
            ochlv = CCXT_Data_Fetcher(
                exchange_id = 'binance',
                symbol = 'XRP/BTC',
                timeframe = '1d',
                candle_amount = 2000,
                save_path = '/content/drive/My Drive/
                ',
                save_format = 'csv'
            )
            candles = ochlv.fetch_candles()
            print(candles.head())
    """

    # Default Values
    def __init__(self,
                 exchange_id = 'binance',
                 symbol = 'BTC/USDT',
                 timeframe = '1d',
                 candle_amount = 500,
                 save_path = '', 
                 save_format = 'csv'):

        self.path = save_path

        self.save_format = save_format
        self.csv = False
        self.sqlite = False
        if self.path != '':
            if self.save_format.lower() == 'csv':
                self.csv = True
            elif self.save_format.lower() == 'sqlite':
                self.sqlite = True

        self.exchange_id = exchange_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.candle_amount = candle_amount

        self.timedeltas_timeframe_suffixes = {
            "s": timedelta(seconds=1),
            "m": timedelta(minutes=1),
            "h": timedelta(hours=1),
            "d": timedelta(days=1),
            "w": timedelta(days=7)
        }

    # Main function
    def fetch_candles(self):
        # Init CCXT exchange object
        self.exchange = getattr(ccxt, self.exchange_id)({
            'enableRateLimit': True
        })

        self.exchange.load_markets() # Requisite CCXT step
        all_symbols = [symbol for symbol in self.exchange.symbols] # Get all valid symbols on exchange
        all_timeframes = [tf for tf in self.exchange.timeframes] # Get all valid timeframes

        self.timeframe = self.timeframe if self.timeframe in all_timeframes else None
        self.symbol = self.symbol if self.symbol in all_symbols else None
        
        # Skip to next symbol if not found on exchange
        if not self.symbol: 
            print("[ERROR] Unsupported symbol {} for exchange {}.".format(self.symbol, exchange_id))
            return None
        if not self.timeframe: # Skip if TF not available on symbol
            print("[ERROR] Unsupported timeframe {} for {}.".format(self.timeframe, exchange_id))
            return None


        print("-- Fetching {} candles for {}".format(self.timeframe, self.symbol))

        # Grab most recent timestamp if data exists already
        if type(self.candle_amount) != str:
            if self.candle_amount > 0:
                since = datetime.utcnow() - (self.candle_amount * self._timedelta(self.timeframe))
            else:
                since = datetime.utcnow() - (500 * self._timedelta(self.timeframe))
        elif self.candle_amount.lower() == 'all':
            since = datetime(1970, 1, 1, tzinfo=timezone.utc) # Earliest possible

        main_path = self.exchange.id + '/' + self.symbol.replace('/','_') + '_' + self.timeframe
        
        if self.csv:
            self.path_to_db_file = self.path + 'csv/' + main_path + '.csv'
            self.path = self.path + 'csv/' + self.exchange.id + '/'
        elif self.sqlite:
            self.path_to_db_file = self.path + 'sqlite/' + main_path + '.sqlite'
            self.path = self.path + 'sqlite/' + self.exchange.id + '/'

        df = pd.DataFrame()
        df_db = pd.DataFrame() # If DB File exists, load it to grab most recent candle timestamp

        # Fetch candles till done
        while True:
            # Can make this more efficient by making it save the timestamp, and load it if else
            if self.path != '' and os.path.exists(self.path_to_db_file):
                #print("\t\t-- Loading existing history from file {} to get next timestamp.".format(path_to_db_file))
                if self.csv:
                    df_db = pd.read_csv(self.path_to_db_file)
                if self.sqlite:
                    conn = self.load_sqlite_db(self.path_to_db_file)
                    if conn:
                        df_db = self.sqlite_to_dataframe(conn, table='ohlcv')
                        
            # Get Latest Candle Timestamp
            if not df_db.empty:
                since = datetime.fromtimestamp(df_db.timestamp.values[-1], timezone.utc)

            # Check if candle DB is up to date
            next_open_date = self.compute_end_timestamp(since, self.timeframe) + self._timedelta('1s')
            if datetime.now(tz=timezone.utc) < next_open_date:
                print("\t-- The time is {} and next candle time is {}, no request needed.".format(exchange_time, since + self._timedelta(self.timeframe)))
                continue

            # Fetch OHLCV data from CCXT
            print("\t-- Fetching candles from {}".format(since.strftime('%m/%d/%Y, %H:%M:%S')))

            df = self.exchange.fetch_ohlcv(symbol=self.symbol, timeframe=self.timeframe, since=int(since.timestamp()*1000))
            df = pd.DataFrame(data=df, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = df['timestamp'].apply(lambda t: int(t/1000)) # convert timestamp from nanoseconds to milliseconds (expected by datetime)

            dupes = 0
            if not df_db.empty and not df.empty:
                # If timstamps from the new list are found in the main db
                dupes_list = df['timestamp'].isin(df_db['timestamp'])
                dupes = len([b for b in dupes_list.values if b == True])

            if df.empty or dupes > 1 or dupes == len(df.timestamp.values):
                print("\t\t-- No new candles received for timeframe: {}, work is done.".format(self.timeframe))
                break
            #else:
            print("\t\t-- {} candles received.".format(len(df)))

            # Writing data to DB format of choice
            if not df_db.empty:
                df = df_db.append(df)
                df.drop_duplicates('timestamp', inplace=True)

            if self.csv:
                self.ifNotDirExists_MakeDir(self.path)
                # Return a csv formatted string to write to .csv file
                data = df.to_csv(mode='a', header=True, index=False)
                # aiofiles is an IO lib compatible with async code
                # w+ creates the file if it doesnt exist
                with open(self.path_to_db_file, 'w+') as f:
                    f.write(data)
            elif self.sqlite:
                self.ifNotDirExists_MakeDir(self.path)
                conn = self.load_sqlite_db(self.path_to_db_file)
                if conn:
                    #print(df.head())
                    c = conn.cursor()
                    c.execute('create table if not exists ohlcv (timestamp, open, high, low, close, volume)')
                    conn.commit()
                    # Use Pandas to write to sqlite db
                    df.to_sql('ohlcv', conn, if_exists='append', index=False) #use index=True if index is timestamp
                    
                    #test writing success
                    c.execute('select * from ohlcv')
                    for row in c.fetchall():
                        print(row)
                    conn.close()
                    
            df_db = df.copy()
            df = pd.DataFrame()

            # Update last candle date
            since = datetime.fromtimestamp(df_db.timestamp.values[-1], timezone.utc)
            time.sleep(self.exchange.rateLimit * 5 / 1000) # soften IO Load

        # Format OHCLV Data for the TensorTrade DataFeed
        df_db = df_db.rename({"timestamp": "Date"}, axis='columns')
        df_db['Date'] = df_db['Date'].apply(lambda x: datetime.utcfromtimestamp(x))
        df_db['Date'] = df_db['Date'].dt.strftime('%Y-%m-%d %I:%M %p')
        df_db.sort_values(by='Date', ascending=True, inplace=True)
        df_db = df_db.set_index("Date")

        # format column names for tensortrade
        if self.exchange.id != 'bitmex' and '/' in self.symbol:
            base, quote = self.symbol.split('/')
        else:
            base = self.symbol[:3]
        df_db.columns = [base + ":" + name.lower() for name in df_db.columns]
        
        print('\t\t\t-- Total Candles: ' + str(len(df_db)) + '\n')
        return df_db

    ####################
    # Helper Functions #
    ####################

    def load_sqlite_db(self, db_file_path):
        conn = None
        try:
            conn = sqlite3.connect(db_file_path)
            return conn
        except Error as e:
            print(e)
        return conn

    def sqlite_to_dataframe(self, conn, table):
        df = pd.read_sql_query(f"select * from {table}", conn)
        return df

    # Converts a timeframe to a datetime.timedelta object
    def _timedelta(self, timeframe):
        for suffix in self.timedeltas_timeframe_suffixes.keys():
            if timeframe.endswith(suffix):
                _ = timeframe.split(suffix)
                c = int(_[0])
                return c * self.timedeltas_timeframe_suffixes[suffix]
        raise RuntimeError("Unable to convert timeframe {} to a fixed timedelta.".format(timeframe))

    def compute_end_timestamp(self, since, timeframe):
        if timeframe == "1M":
            # Special case for month because it has not fixed timedelta
            return datetime(since.year, since.month, 1, tzinfo=timezone.utc) - self._timedelta('1s')

        td = self._timedelta(timeframe)
        # This is equivalent to the timestamp
        start_of_current_bar = int(since.timestamp() / td.total_seconds()) * td.total_seconds()
        return datetime.fromtimestamp(start_of_current_bar, timezone.utc) - self._timedelta('1s')

    def ifNotDirExists_MakeDir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
