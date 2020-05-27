# -*- coding: utf-8 -*-
"""
Use this URL for a Google Colab Demo of this class and its usage:
    https://colab.research.google.com/drive/154_2tvDn_36pZzU_XkSv9Xvd3KjQCw1U
"""
from datetime import timedelta, datetime, timezone
import sys, os, time, random
import pandas as pd

import json
import csv
import sqlite3
from sqlite3 import Error

import ccxt
import yfinance as yf

class Data():
    """ 
        Class Wraps CCXT and Yahoo Finance Data Fetching Functions
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

    class CCXT():
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
                        since = datetime.utcnow() - (c * candle_amount * Data._timedelta(timeframe))
                        since = datetime(since.year, since.month, 1, tzinfo=timezone.utc)
                    else:
                        since = datetime.utcnow() - (candle_amount * Data._timedelta(timeframe))
            elif candle_amount.lower() == 'all':
                since = datetime(1970, 1, 1, tzinfo=timezone.utc)
            else:
                if timeframe.endswith('M'):
                    since = datetime(1970, 1, 1, tzinfo=timezone.utc)
                else:
                    since = datetime.utcnow() - (500 * Data._timedelta(timeframe))

            since = Data._earliest_datetime(since) # sanitize if date is earlier than 1970

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
                        conn = Data.Save_Data._load_sqlite_db(path_to_db_file)
                        if conn:
                            df_db = Data._sqlite_to_dataframe(conn, table='ohlcv')

                if not df_db.empty:
                    since = datetime.fromtimestamp(df_db.timestamp.values[-1], timezone.utc) # said close tiemstamp before, not open

                # Check if candle DB is up to date
                next_open_date = Data._compute_end_timestamp(since, timeframe) + Data._timedelta('1s')
                if datetime.now(tz=timezone.utc) < next_open_date:
                    print("\t-- The next candle time is {}, no request needed.".format(since + Data._timedelta(timeframe)))
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
                    Data.Save_Data.as_csv(df, path_to_db_file)
                elif sqlite:
                    sql_query = 'create table if not exists ohlcv (timestamp, open, high, low, close, volume)'
                    sql_table_name = 'ohlcv'     
                    Data.Save_Data.as_sqlite(df, path_to_db_file, sql_table_name, sql_query)

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
                    conn = Save_Data._load_sqlite_db(path_to_db_file)
                    if conn:
                        df_db = Data._sqlite_to_dataframe(conn, table='trades')
            else:
                df_db = pd.DataFrame() # If DB File exists, load it to grab most recent candle timestamp
            
            prev_df_len = len(df_db) # Useful to stop endless loop later

            # Grab most recent timestamp if data exists already
            # Else set a default start date
            if trades_amount != 'all':
                since = datetime.utcnow() - Data._timedelta(trades_amount)
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
                        df = Data._ticks_to_df(df, exchange_id)

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
                new_since_date = datetime.fromtimestamp(df_db.timestamp.values[-1]/1000, timezone.utc) if len(df_db) > 0 else since + Data._timedelta(trades_amount)  

                print("\t\t-- {} trades received.".format(len(df)))

                # Save Data to DB file
                if csv:
                    Save_Data.as_csv(df, path_to_db_file)
                elif sqlite:
                    if ccxt_exchange.id == 'bitmex':
                       sql_query = """
                            create table if not exists trades 
                            (timestamp, datetime, symbol, order, type, side, price, amount, homeNotional, foreignNotional, id)
                        """
                    else:
                        sql_query = """
                            create table if not exists trades 
                            (timestamp, datetime, symbol, order, type, side, price, amount, cost, id)
                        """
                    sql_table_name = 'trades'     
                    Save_Data.as_sqlite(df, path_to_db_file, sql_table_name, sql_query)

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

    class Yahoo():
        """               
            symbol: Can be multiple symbols separated by a space.
                    eg. 'AAPL MSFT AMZN SNAP SPY' will get all of those

            timeframe: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
                       (optional, default is '1d')
            
            amount: Use 'amount' instead of start or end
                    Use 'all' or 'max' to get FULL candle history. 
                    valid amounts: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max, all
                    (this is an optional parameter, default is '1mo')
            
            save_path: Use if you want to save data as .csv file or SQLite DB.
            save_format: 'csv' or 'sqlite' are the database options.
            
            TT_Format: True would set columns with a prefix using the base symbol
                eg. BTC:open, BTC:close, BTC:volume


            Example Usage:
                from tensortrade.utils import Data

                # Get 1 Day candles from 2015 to 2020
                Data.Yahoo.fetch_candles(symbol="SPY AAPL MSFT",
                                         start="2015-01-01",
                                         end="2020-01-01",
                                         timeframe="1d")


                # Get 1 Week candles from 2015 to 2020
                Data.Yahoo.fetch_candles(symbol="SPY AAPL MSFT",
                                         start="2015-01-01",
                                         candle_amount="5y",
                                         timeframe="1wk")

                # Get full asset history of 1 Month candles
                Data.Yahoo.fetch_candles(symbol="SPY AAPL MSFT",
                                         candle_amount="max",
                                         timeframe="1mo")
        """

        @classmethod
        def fetch_candles(cls,
                          symbol: str = 'SPY',
                          timeframe: str = '1d',
                          start = None,
                          end = None,
                          candle_amount: str = '5y',
                          save_path = '', 
                          save_format: str = 'csv',
                          TT_Format=False):
            """
                Fetch OHLCV aka Candle Data using yfinance
                Able to fetch full candle history
                Options to save to CSV or SQLite DB files
            """

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

            symbol = symbol.upper()
            candle_amount = candle_amount.lower() if candle_amount else 'max' if candle_amount.lower() == 'all' else None

            all_timeframes = ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo']
            timeframe = timeframe.lower() if timeframe.lower() in all_timeframes else None

            if not timeframe: # Skip if TF not available
                print("[ERROR] Unsupported timeframe {} for Yahoo Finance.".format(timeframe))
                return None

            print("-- Fetching {} candles for {}".format(timeframe, symbol))

            retries = 3
            while retries > 0:
                try:   
                    df = yf.download(
                        tickers = symbol,
                        start = start,
                        end = end,
                        period = candle_amount,
                        interval = timeframe,
                        auto_adjust = True,
                        group_by = 'ticker',
                        prepost = True
                    )
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


            for asset in symbol.split(' '):
                main_path = 'Yahoo Finance/' + asset + '_' + timeframe
                if csv:
                    path_to_db_file = path + 'csv/' + main_path + '.csv'
                    path = path + 'csv/'
                elif sqlite:
                    path_to_db_file = path + 'sqlite/' + main_path + '.sqlite'
                    path = path + 'sqlite/'

                df_db = pd.DataFrame() # If DB File exists, load it
                if path and os.path.exists(path_to_db_file):
                    #print("\t\t-- Loading existing history from file {} to get next timestamp.".format(path_to_db_file))
                    try:
                        if csv:
                            df_db = pd.read_csv(path_to_db_file)
                        if sqlite:
                            conn = Data._load_sqlite_db(path_to_db_file)
                            if conn:
                                df_db = Data._sqlite_to_dataframe(conn, table='ohlcv')
                    except Exception as e:
                        print(e)

                # Writing data to DB format of choice
                if not df_db.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df_db = df_db.append(df[asset])
                    else:
                        df_db = df_db.append(df)

                    df_db = df_db.reset_index().drop_duplicates('Datetime')
                    #df_db['Date'] = df_db['Date'].apply(lambda x: pd.to_datetime(x))
                    df_db.sort_values(by='Datetime', ascending=True, inplace=True)
                    df_db = df_db.set_index('Datetime')

                    if isinstance(df.columns, pd.MultiIndex):
                        df[asset] = df_db.copy()
                    else:
                        df = df_db.copy()

                # Save Data to DB file
                if len(df.index.values) > 0:
                    if csv:
                        Data.Save_Data.as_csv(df.reset_index(), path_to_db_file)
                    elif sqlite:
                        sql_query = 'create table if not exists ohlcv (date, open, high, low, close, volume)'
                        sql_table_name = 'ohlcv'     
                        Data.Save_Data.as_sqlite(df.reset_index(), path_to_db_file, sql_table_name, sql_query)

            # After getting all the candles
            print('\t\t\t-- Total Candles: ' + str(len(df)) + '\n')
            # Format OHCLV Data for the TensorTrade DataFeed
            if TT_Format:
                if isinstance(df.columns, pd.MultiIndex):
                    df_new = pd.DataFrame()
                    for asset in symbol.split(' '):
                        df_copy = df[asset].copy()
                        df_copy.columns = [asset + ":" + name.lower() for name in df[asset].columns]
                        df_new = pd.concat([df_new, df_copy],axis=1)
                    df = df_new
                else:
                    df.columns = [symbol + ":" + name.lower() for name in df.columns]
            else:
                df.columns = [name.lower() for name in df.columns]
                
            if 'volume' in df.columns:
                df['volume'] = df['volume'].apply(lambda x: float(x))
               
            return df


    class Save_Data():
        """ Saves Data as CSV or SQLite """

        @classmethod
        def _ifNotDirExists_MakeDir(cls, path):
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

        @classmethod
        def _load_sqlite_db(cls, db_file_path):
            conn = None
            try:
                conn = sqlite3.connect(db_file_path)
                return conn
            except Error as e:
                print(e)
            return conn

        @classmethod
        def as_csv(cls, df, path_to_db_file):
            cls._ifNotDirExists_MakeDir(path_to_db_file[:path_to_db_file.rfind('/')])
            # Return a csv formatted string to write to .csv file
            data = df.to_csv(mode='a', header=True, index=False)
            # w+ creates the file if it doesnt exist
            with open(path_to_db_file, 'w+') as f:
                f.write(data)
            print(f'Data written to CSV @ {path_to_db_file}')

        @classmethod
        def as_sqlite(cls, df, path_to_db_file, sql_table_name, sql_query):
            cls._ifNotDirExists_MakeDir(path_to_db_file[:path_to_db_file.rfind('/')])
            conn = cls._load_sqlite_db(path_to_db_file)
            if conn:
                c = conn.cursor()
                c.execute(sql_query)
                conn.commit()
                # Use Pandas to write to sqlite db
                df.to_sql(sql_table_name, conn, if_exists='append', index=False) #use index=True if index is timestamp
                
                #test writing success
                c.execute(f'select * from {sql_table_name}')
                for row in c.fetchall():
                    print(row)

                conn.close()
    ####################
    # Helper functions #
    ####################

    def _earliest_datetime(dt):
        if dt.timestamp() < 0.0:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)
        else:
            return dt

    def _resample_ticks(data, column, tf):
        # If 'price' in column name
        if column.find('price') > 0 or column == 'price' or 'price' in column:
            return data[column].resample(tf).ohlc()
        else: # Resample Volume
            return data[column].resample(tf).sum()

    def _ticks_to_df(ticks, exchange_id):
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

    def _sqlite_to_dataframe(conn, table):
        df = pd.read_sql_query(f"select * from {table}", conn)
        return df

    # Convert a timeframe to a datetime.timedelta object
    def _timedelta(timeframe):
        for suffix in Data.timedeltas_timeframe_suffixes.keys():
            if timeframe.endswith(suffix):
                _ = timeframe.split(suffix)
                c = int(_[0])
                return c * Data.timedeltas_timeframe_suffixes[suffix]
        raise RuntimeError("Unable to convert timeframe {} to a fixed timedelta.".format(timeframe))

    def _compute_end_timestamp(since, timeframe):
        if timeframe == "1M":
            # Special case for month because it has not fixed timedelta
            return datetime(since.year, since.month, 1, tzinfo=timezone.utc) - Data._timedelta('1s')

        td = Data._timedelta(timeframe)
        # This is equivalent to the timestamp
        start_of_current_bar = int(since.timestamp() / td.total_seconds()) * td.total_seconds()
        return datetime.fromtimestamp(start_of_current_bar, timezone.utc) - Data._timedelta('1s')
