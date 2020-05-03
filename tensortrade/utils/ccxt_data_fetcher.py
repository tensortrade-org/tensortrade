# -*- coding: utf-8 -*-
"""
Use this URL for a Google Colab Demo of this class and its usage:
    https://colab.research.google.com/drive/1h9SnR2aqn3cuyoew4QdxhKjTe1VjgxUr
"""

import ccxt

from datetime import timedelta, datetime, timezone
import sys, os, time, random
import pandas as pd
import json

import csv
import sqlite3
from sqlite3 import Error

class CCXT_Data():
    """
        The majority of code credit goes to: 
            https://github.com/Celeborn2BeAlive/cryptobigbro
            
        exchange_id: Any exchange id available thru CCXT 
                     https://github.com/ccxt/ccxt/wiki/Manual#exchanges

        symbol: A slash is used for all symbols except on BitMEX Futures.
                eg. XRPH20 has no slash,
                but BTC/USD and ETH/USD are how they identify the USD pairs.

        timeframe: Any timeframe available on the chosen exchange.
        
        candle_amount: Use 'all' to get FULL candle history. 
                       Default is 500.
                       
        trades_amount: Use 'all' to get FULL trade history. 
                       Default is '10m' aka 10 mins.
        
        save_path: Use if you want to save data as .csv file or SQLite DB.
        save_format: 'csv' or 'sqlite' are the database options.
        
        TT_Format: True would set columns with a prefix using the base symbol
            eg. BTC:open, BTC:close, BTC:volume

        Example Usage:
            from tensortrade.utils.ccxt_data_fetcher import CCXT_Data
            
            # Fetch Trades
            trades = CCXT_Data.fetch_trades(
                exchange = 'bitmex', 
                symbol = 'BTC/USD', 
                trades_amount = '10m', ## Get 10 minutes worth of trades
                save_path = '/content/drive/My Drive/',
                save_format = 'csv' 
            )
            
            # Fetch Candles
            ohlcv = CCXT_Data.fetch_candles(
                exchange = 'binance', 
                symbol = 'BTC/USDT',
                timeframe = '1d',
                candle_amount = '1000', ## Get 1000 1 Day candles
                save_path = '/content/drive/My Drive/Crypto_SQLite_DBs/', 
                save_format = 'sqlite' 
            )
    """

    timedeltas_timeframe_suffixes = {
        "s": timedelta(seconds=1),
        "m": timedelta(minutes=1),
        "h": timedelta(hours=1),
        "d": timedelta(days=1),
        "w": timedelta(days=7),
        "M": timedelta(days=31),
        "Y": timedelta(weeks=52), # option for fetch trades
        "y": timedelta(weeks=52) # lowercase alias
    }

    @classmethod
    def fetch_candles(cls, 
                      exchange: str = 'binance',
                      symbol: str = 'BTC/USDT',
                      timeframe: str = '1d',
                      candle_amount: int = 1000,
                      save_path = '', 
                      save_format: str = 'csv',
                      limit: int = 1000,
                      TT_Format=False):
        """
            Fetch OHLCV aka Candle Data using CCXT
            Able to fetch full available candle history
            Options to save to CSV or SQLite DB files
        """

        mk_path = ''
        path = save_path
        path_to_db_file = ''

        csv = False
        sqlite = False
        if path:
            if save_format.lower() == 'csv':
                csv = True
            if save_format.lower() == 'sqlite':
                sqlite = True

        exchange_id = exchange.lower()
        symbol = symbol.upper()


        # Init CCXT exchange object
        ccxt_exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True
        })

        ccxt_exchange.load_markets() # Requisite CCXT step
        all_symbols = [symbol for symbol in ccxt_exchange.symbols] # Get all valid symbols on exchange
        all_timeframes = [tf for tf in ccxt_exchange.timeframes] # Get all valid timeframes

        timeframe = timeframe if timeframe in all_timeframes else None
        symbol = symbol if symbol in all_symbols else None
        
        # Skip to next symbol if not found on exchange
        if not symbol: 
            print("[ERROR] Unsupported symbol {} for exchange {}.".format(symbol, exchange_id))
            return None
        if not timeframe: # Skip if TF not available on symbol
            print("[ERROR] Unsupported timeframe {} for {}.".format(timeframe, exchange_id))
            return None


        print("-- Fetching {} candles for {}".format(timeframe, symbol))

        # Grab most recent timestamp if data exists already
        if type(candle_amount) != str:
            if candle_amount > 0:
                if timeframe.endswith('M'):
                    _ = timeframe.split('M')
                    c = int(_[0])
                    # Special case for month because it has not fixed timedelta
                    since = datetime.utcnow() - (c * candle_amount * cls._timedelta(timeframe))
                    since = datetime(since.year, since.month, 1, tzinfo=timezone.utc)
                else:
                    since = datetime.utcnow() - (candle_amount * cls._timedelta(timeframe))
        elif candle_amount.lower() == 'all':
            since = datetime(1970, 1, 1, tzinfo=timezone.utc)
        else:
            if timeframe.endswith('M'):
                since = datetime(1970, 1, 1, tzinfo=timezone.utc)
            else:
                since = datetime.utcnow() - (500 * cls._timedelta(timeframe))

        since = cls.earliest_datetime(since) # sanitize if date is earlier than 1970

        main_path = ccxt_exchange.id + '/' + symbol.replace('/','_') + '_' + timeframe
        if csv:
            path_to_db_file = path + 'csv/' + main_path + '.csv'
            mk_path = path + 'csv/'
            path = path + 'csv/' + ccxt_exchange.id + '/'
        elif sqlite:
            path_to_db_file = path + 'sqlite/' + main_path + '.sqlite'
            mk_path = path + 'sqlite/'
            path = path + 'sqlite/' + ccxt_exchange.id + '/'

        df = pd.DataFrame()
        df_db = pd.DataFrame() # If DB File exists, load it to grab most recent candle timestamp

        # Fetch candles till done
        while True:
            # Can make this more efficient by making it save the timestamp, and load it if else
            if path and os.path.exists(path_to_db_file):
                #print("\t\t-- Loading existing history from file {} to get next timestamp.".format(path_to_db_file))
                if csv:
                    df_db = pd.read_csv(path_to_db_file)
                if sqlite:
                    conn = cls.load_sqlite_db(path_to_db_file)
                    if conn:
                        df_db = cls.sqlite_to_dataframe(conn, table='ohlcv')

            if not df_db.empty:
                since = datetime.fromtimestamp(df_db.timestamp.values[-1], timezone.utc) # said close tiemstamp before, not open

            # Check if candle DB is up to date
            next_open_date = cls.compute_end_timestamp(since, timeframe) + cls._timedelta('1s')
            if datetime.now(tz=timezone.utc) < next_open_date:
                print("\t-- The next candle time is {}, no request needed.".format(since + cls._timedelta(timeframe)))
                continue

            # Fetch candle data with CCXT
            print("\t-- Fetching candles from {}".format(since.strftime('%m/%d/%Y, %H:%M:%S')))
            retries = 3
            while retries > 0:
                try:
                    df = ccxt_exchange.fetch_ohlcv(symbol=symbol, 
                                                   timeframe=timeframe, 
                                                   since=int(since.timestamp()*1000),
                                                   limit=limit)
                    df = pd.DataFrame(data=df, columns=['timestamp','open','high','low','close','volume'])
                    df['timestamp'] = df['timestamp'].apply(lambda t: int(t/1000)) # convert timestamp from nanoseconds to milliseconds (expected by datetime)
                    break
                except Exception as error:
                    if retries == 3:
                        print('Retry 1/3 | Got an error', type(error).__name__, error.args, ', retrying in 3 seconds.')
                        time.sleep(3)
                    elif retries == 2:
                        print('Retry 2/3 | Got an error', type(error).__name__, error.args, ', retrying in 10 seconds...')
                        time.sleep(10)
                    else:
                        print('Final Retry: Got an error', type(error).__name__, error.args, ', retrying in 25 seconds...')
                        time.sleep(25)
                    retries -= 1
            
            #else:
            print("\t\t-- {} candles received.".format(len(df)))

            # Writing data to DB format of choice
            if not df_db.empty:
                df = df_db.append(df)
                df.drop_duplicates('timestamp', inplace=True)

            # Save Data to DB file
            if csv:
                cls.ifNotDirExists_MakeDir(mk_path)
                cls.ifNotDirExists_MakeDir(path)
                # Return a csv formatted string to write to .csv file
                data = df.to_csv(mode='a', header=True, index=False)
                # aiofiles is an IO lib compatible with async code
                # w+ creates the file if it doesnt exist
                with open(path_to_db_file, 'w+') as f:
                    f.write(data)
            elif sqlite:
                cls.ifNotDirExists_MakeDir(mk_path)
                cls.ifNotDirExists_MakeDir(path)
                conn = cls.load_sqlite_db(path_to_db_file)
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

            new_since_date = df.timestamp.values[-1] # Most recent timestamp in db

            if df.empty or since.timestamp() == new_since_date:
                print("\t\t-- No new candles received for timeframe: {}, work is done.".format(timeframe))
                break

            df_db = df.copy()
            df = pd.DataFrame()

            # Update last candle date
            since = datetime.fromtimestamp(df_db.timestamp.values[-1], timezone.utc)
            #time.sleep(ccxt_exchange.rateLimit * 5 / 1000) # soften IO Load

        # After getting all the candles, format and return for tensortrade use
        # Format OHCLV Data for the TensorTrade DataFeed
        df_db.sort_values(by='timestamp', ascending=True, inplace=True)

        if TT_Format:
            df_db = df_db.rename({"timestamp": "Date"}, axis='columns')
            df_db['Date'] = df_db['Date'].apply(lambda x: datetime.utcfromtimestamp(x))
            df_db['Date'] = df_db['Date'].dt.strftime('%Y-%m-%d %H:%M %p')
            df_db = df_db.set_index("Date")

            # Format column names for tensortrade use
            if ccxt_exchange.id != 'bitmex' and '/' in symbol:
                base, quote = symbol.split('/')
            else:
                base = symbol[:3]
            df_db.columns = [base + ":" + name.lower() for name in df_db.columns]
        else:
            df_db = df_db.rename({"timestamp": "Date"}, axis='columns')
            df_db['Date'] = df_db['Date'].apply(lambda x: datetime.utcfromtimestamp(x))
            df_db = df_db.set_index("Date")
        
        print('\t\t\t-- Total Candles: ' + str(len(df_db)) + '\n')

        return df_db

    @classmethod
    def fetch_trades(cls, 
                     exchange: str = 'bitmex',
                     symbol: str = 'BTC/USD',
                     trades_amount: str = '10m',
                     save_path: str = '', 
                     save_format: str = 'csv',
                     limit: int=1000,
                     TT_Format=False):
        """
            Fetch Trades aka Tick Data using CCXT
            Able to fetch full available trade history
            Options to save to CSV or SQLite DB files
            resample_ticks() converts trades to any candle timeframe
        """

        mk_path = ''
        path = save_path
        path_to_db_file = ''
        since = None

        csv = False
        sqlite = False
        if path:
            if save_format.lower() == 'csv':
                csv = True
            if save_format.lower() == 'sqlite':
                sqlite = True

        exchange_id = exchange.lower()
        symbol = symbol.upper()

        # Init CCXT exchange object
        ccxt_exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True
        })

        ccxt_exchange.load_markets() # Requisite CCXT step
        all_symbols = [symbol for symbol in ccxt_exchange.symbols] # Get all valid symbols on exchange
        symbol = symbol if symbol in all_symbols else None
        
        # Skip to next symbol if not found on exchange
        if not symbol: 
            print("[ERROR] Unsupported symbol {} for exchange {}.".format(symbol, exchange_id))
            return None

        print("-- Fetching Trades for {}".format(symbol))

        main_path = ccxt_exchange.id + '/' + symbol.replace('/','_') + '_Tick_Data'
        if csv:
            path_to_db_file = path + 'csv/' + main_path + '.csv'
            mk_path = path + 'csv/'
            path = path + 'csv/' + ccxt_exchange.id + '/'
        elif sqlite:
            path_to_db_file = path + 'sqlite/' + main_path + '.sqlite'
            mk_path = path + 'sqlite/'
            path = path + 'sqlite/' + ccxt_exchange.id + '/'

        # Load previous DB if exists
        # Can make this more efficient by making it save the timestamp, and load it if else
        if path and os.path.exists(path_to_db_file):
            if csv:
                df_db = pd.read_csv(path_to_db_file)
            if sqlite:
                conn = cls.load_sqlite_db(path_to_db_file)
                if conn:
                    df_db = cls.sqlite_to_dataframe(conn, table='trades')
        else:
            df_db = pd.DataFrame() # If DB File exists, load it to grab most recent candle timestamp
        
        prev_df_len = len(df_db) # Useful to stop endless loop later

        # Grab most recent timestamp if data exists already
        # Else set a default start date
        if trades_amount != 'all':
            since = datetime.utcnow() - cls._timedelta(trades_amount)
        elif trades_amount.lower() == 'all': #or cls.since 
            # check if since can be converted to datetime
            # try to conver it with default format, and failing that
            # except: shove it into datetime 
            since = datetime(1970, 1, 1, tzinfo=timezone.utc) # Earliest possible

        endless_loop_protection = 0
        # Fetch trades till done
        while True:
            df = pd.DataFrame()

            if not df_db.empty:
                since = datetime.fromtimestamp(df_db.timestamp.values[-1]/1000, timezone.utc) # said close tiemstamp before, not open

            # Fetch Tick data from CCXT
            print("\t-- Fetching Trades since {}".format(since.strftime('%m/%d/%Y, %H:%M:%S.%f')))

            retries = 3
            while retries > 0:
                try:
                    df = ccxt_exchange.fetch_trades(symbol=symbol,
                                               since=int(since.timestamp()*1000),
                                               limit=limit)
                    # Convert Dictionary of Arrays/Lists into a DataFrame
                    df = cls.ticks_to_df(df, exchange_id)

                    # Check for duplicates
                    dupes = 0
                    if not df_db.empty and not df.empty:
                        # If timestamps from the new list are found in the main db
                        dupes_list = df['id'].isin(df_db['id'])
                        dupes = len([b for b in dupes_list.values if b == True])
                        df_db = pd.concat([df_db, df], axis=0)
                    else:
                        df_db = df.copy()
                    
                    df_db.drop_duplicates('id', inplace=True)
                    break
                except Exception as error:
                    if retries == 3:
                        print('Retry 1/3 | Got an error', type(error).__name__, error.args, ', retrying in 3 seconds.')
                        time.sleep(3)
                    elif retries == 2:
                        print('Retry 2/3 | Got an error', type(error).__name__, error.args, ', retrying in 10 seconds...')
                        time.sleep(10)
                    elif retries == 1:
                        print('Final Retry: Got an error', type(error).__name__, error.args, ', retrying in 15 seconds...')
                        time.sleep(15)
                    else:
                        return df_db
                    retries -= 1

            # Break from while True loop if no more data to get
            if len(df) < int(prev_df_len/10):
                endless_loop_protection += 1 # Detect when are just getting live trades
            else:
                prev_df_len = len(df)

            # Most recent timestamp in db
            new_since_date = datetime.fromtimestamp(df_db.timestamp.values[-1]/1000, timezone.utc) if len(df_db) > 0 else since + cls._timedelta(trades_amount)  

            print("\t\t-- {} trades received.".format(len(df)))
               
            # Save Data to DB file
            if csv:
                cls.ifNotDirExists_MakeDir(mk_path)
                cls.ifNotDirExists_MakeDir(path)
                # Return a csv formatted string to write to .csv file
                data = df_db.to_csv(mode='a', header=True, index=False)
                # aiofiles is an IO lib compatible with async code
                # w+ creates the file if it doesnt exist
                with open(path_to_db_file, 'w+') as f:
                    f.write(data)
            elif sqlite:
                cls.ifNotDirExists_MakeDir(mk_path)
                cls.ifNotDirExists_MakeDir(path)
                conn = cls.load_sqlite_db(path_to_db_file)
                if conn:
                    #print(df.head())
                    c = conn.cursor()
                    if ccxt_exchange.id == 'bitmex':
                        """
                        create table if not exists trades 
                        (timestamp, datetime, symbol, order, type, side, price, amount, homeNotional, foreignNotional, id)
                        """
                    else:
                        """
                        create table if not exists trades 
                        (timestamp, datetime, symbol, order, type, side, price, amount, cost, id)
                        """

                    c.execute(SQL_string)

                    conn.commit()
                    # Use Pandas to write to sqlite db
                    df_db.to_sql('trades', conn, if_exists='append', index=False) #use index=True if index is timestamp
                    
                    #test writing success
                    c.execute('select * from trades')
                    for row in c.fetchall():
                        print(row)

                    conn.close()
            print(since.timestamp(), new_since_date.timestamp())
            if df.empty or since.timestamp() == new_since_date.timestamp() or endless_loop_protection >= 5:
                print("\t\t-- Stopping. Work is done.")
                break
            #else:
            # Update last candle date
            since = new_since_date
            #time.sleep(ccxt_exchange.rateLimit * 5 / 1000) # soften IO Load

        # After getting all the trades/ticks, format and return for tensortrade use
        # Format Tick Data for the TensorTrade DataFeed
        df_db.sort_values(by='timestamp', ascending=True, inplace=True)
        df_db['Date'] = df_db['timestamp'].apply(lambda x: datetime.fromtimestamp(x/1000, timezone.utc))
        df_db = df_db.set_index("Date")

        if TT_Format:
            # format column names for tensortrade
            if ccxt_exchange.id != 'bitmex' and '/' in symbol:
                base, quote = symbol.split('/')
            else:
                base = symbol[:3]
            df_db.columns = [base + ":" + name.lower() for name in df_db.columns]
        
        print('\t\t\t-- Total Trades: ' + str(len(df_db)) + '\n')
        return df_db




    ####################
    # Helper functions #
    ####################

    @classmethod
    def earliest_datetime(cls, dt):
        if dt.timestamp() < 0.0:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)
        else:
            return dt

    @classmethod
    def resample_ticks(cls, data, column, tf):
        # If 'price' in column name
        if column.find('price') > 0 or column == 'price' or 'price' in column:
            return data[column].resample(tf).ohlc()
        else: # Resample Volume
            return data[column].resample(tf).sum()

    @classmethod
    def ticks_to_df(cls, ticks, exchange_id):
        # CCXT fetch_trades() data format
        # [{
        # 'info':       { ... },                  // the original decoded JSON as is
        # 'id':        '12345-67890:09876/54321', // string trade id
        # 'timestamp':  1502962946216,            // Unix timestamp in milliseconds
        # 'datetime':  '2017-08-17 12:42:48.000', // ISO8601 datetime with milliseconds
        # 'symbol':    'ETH/BTC',                 // symbol
        # 'order':     '12345-67890:09876/54321', // string order id or undefined/None/null
        # 'type':      'limit',                   // order type, 'market', 'limit' or undefined/None/null
        # 'side':      'buy',                     // direction of the trade, 'buy' or 'sell'
        # 'price':      0.06917684,               // float price in quote currency
        # 'amount':     1.5,                      // amount of base currency
        # }]

        tick_data = {
            'side': [],
            'symbol': [],
            'timestamp': [],
            'datetime': [],
            'price': [],
            'amount': [],
            'id': []
        }
        for i in range(0, len(ticks)):
            tick_data['side'].append(ticks[i]['side'])
            tick_data['symbol'].append(ticks[i]['symbol'])
            tick_data['timestamp'].append(ticks[i]['timestamp'])
            tick_data['datetime'].append(ticks[i]['datetime'])
            tick_data['price'].append(ticks[i]['price'])
            tick_data['amount'].append(ticks[i]['amount'])
            tick_data['id'].append(ticks[i]['id'])

            if exchange_id == 'bitmex':
                if i == 0:
                    tick_data['homeNotional'] = []
                    tick_data['foreignNotional'] = []
                tick_data['homeNotional'].append(ticks[i]['info']['homeNotional'])
                tick_data['foreignNotional'].append(ticks[i]['info']['foreignNotional'])
            else:
                if i == 0:
                    tick_data['cost'] = []
                tick_data['cost'].append(ticks[i]['cost'])

        return pd.DataFrame(tick_data)

    # SQLite
    @classmethod
    def load_sqlite_db(cls, db_file_path):
        conn = None
        try:
            conn = sqlite3.connect(db_file_path)
            return conn
        except Error as e:
            print(e)
        return conn

    @classmethod
    def sqlite_to_dataframe(cls, conn, table):
        df = pd.read_sql_query(f"select * from {table}", conn)
        return df

    # Convert a timeframe to a datetime.timedelta object
    @classmethod
    def _timedelta(cls, timeframe):
        for suffix in cls.timedeltas_timeframe_suffixes.keys():
            if timeframe.endswith(suffix):
                _ = timeframe.split(suffix)
                c = int(_[0])
                return c * cls.timedeltas_timeframe_suffixes[suffix]
        raise RuntimeError("Unable to convert timeframe {} to a fixed timedelta.".format(timeframe))

    @classmethod
    def compute_end_timestamp(cls, since, timeframe):
        if timeframe == "1M":
            # Special case for month because it has not fixed timedelta
            return datetime(since.year, since.month, 1, tzinfo=timezone.utc) - cls._timedelta('1s')

        td = cls._timedelta(timeframe)
        # This is equivalent to the timestamp
        start_of_current_bar = int(since.timestamp() / td.total_seconds()) * td.total_seconds()
        return datetime.fromtimestamp(start_of_current_bar, timezone.utc) - cls._timedelta('1s')

    @classmethod
    def ifNotDirExists_MakeDir(cls, path):
        if not os.path.exists(path):
            os.mkdir(path)
