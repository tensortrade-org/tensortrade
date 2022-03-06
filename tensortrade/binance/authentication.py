#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File:        tensortrade/binance/authentication.py
# By:          Samuel Duclos
# For:         Myself
# Description: This file handles python-binance and binance_f client authentication.

from os.path import exists
from binance.client import Client
from binance_f import RequestClient

class Binance_authenticator:
    def __init__(self, 
                 spot_client=None, 
                 futures_client=None, 
                 get_spot_client=True, 
                 get_futures_client=False, 
                 keys_path='keys.txt'):

        api_key, secret_key = self.get_API_keys(keys_path=keys_path)
        self.spot_client = Client(api_key=api_key, api_secret=secret_key) if get_spot_client else spot_client
        self.futures_client = RequestClient(api_key=api_key, secret_key=secret_key) if get_futures_client else futures_client

    def get_API_keys(self, keys_path='keys.txt'):
        if exists(keys_path):
            with open(keys_path, 'r') as f:
                return f.readline().replace('\n', '').split(':')
        else:
            return ('', '')

