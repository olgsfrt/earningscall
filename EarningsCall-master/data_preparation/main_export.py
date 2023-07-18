#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:23:36 2019

@author: mschnaubelt
"""

import multiprocessing as mp
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from export_util import create_data_index
from export_adjusted_prices import load_adjusted_prices
from export_fundamental import load_eps, load_events, load_fundamental, \
                                load_marketcap, load_fundratio


BASE_DIR = '/mnt/data/earnings_calls/'

DATA_DIR = BASE_DIR + 'eikon/data/'
SYMBOL_LIST_FILE = BASE_DIR + 'eikon/symbol_list_excel.xlsx'

EXPORT_DIR = BASE_DIR + 'export/'


FUNDAMENTALS = [('revenue', 'Revenue'),
                ('bvps', 'Book Value Per Share'),
                ('cfps', 'Cash Flow Per Share'),
                ('fcfps', 'Free Cash Flow Per Share'),
                ('divest', 'Dividend Per Share'),
                ('roa', 'Return On Assets'),
                ('roe', 'Return On Equity'),
                ('oprofit', 'Operating Profit'),
                
                ('cfo', 'Cash Flow From Operations'),
                ('fcf', 'Free Cash Flow'),
                ('gpm', 'Gross Profit Margin'),
                ('netdebt', 'Net Debt'),
                ('totdebt', 'Total Debt'),
                ('netinc', 'Net Income'),
                ('opex', 'Operating Expense'),
                ('pretaxinc', 'Pre-Tax Profit'),
                ('shequity', 'Shareholders Equity'),
                ('totassets', 'Total Assets'),
                ('nshares', 'Number of Shares Outstanding')
                ]

#file_suffix, eikon_name = FUNDAMENTALS[0]
#data=load_fundamental(DATA_DIR, file_suffix, eikon_name)



def price_pool_worker(index_row):
    print("Processing %s" % index_row.name)
    try:
        data = load_adjusted_prices(DATA_DIR, 
                                    export_key = index_row['sa_ticker'],
                                    primary_ric = index_row['primary_ric'],
                                    rics = index_row['rics'])
        
        data = data[['instrument_key', 'Date', 'Currency', 'Price Open', 'Price Close', 
                     'Volume', 'div', 'Div Currency', 'Factor', 'Open', 'Close']].copy()
        
        return data
    except Exception as e:
        print("Exception while processing symbol", index_row['sa_ticker'])
        print('\t', e)
        return {'sa_ticker': index_row['sa_ticker'],
                'rics': index_row['rics'],
                'item': 'PRICE',
                'error': e}


def marketcap_pool_worker(index_row):
    print("Processing %s" % index_row.name)
    try:
        data = load_marketcap(DATA_DIR, 
                              export_key = index_row['sa_ticker'],
                              primary_ric = index_row['primary_ric'],
                              rics = index_row['rics'])
        
        return data
    except Exception as e:
        print("Exception while processing symbol", index_row['sa_ticker'])
        print('\t', e)
        return {'sa_ticker': index_row['sa_ticker'],
                'rics': index_row['rics'],
                'item': 'MKTCAP',
                'error': e}


def fundratio_pool_worker(index_row):
    print("Processing %s" % index_row.name)
    try:
        data = load_fundratio(DATA_DIR, 
                              export_key = index_row['sa_ticker'],
                              primary_ric = index_row['primary_ric'],
                              rics = index_row['rics'])
        
        return data
    except Exception as e:
        print("Exception while processing symbol", index_row['sa_ticker'])
        print('\t', e)
        return {'sa_ticker': index_row['sa_ticker'],
                'rics': index_row['rics'],
                'item': 'FUNDRATIO',
                'error': e}

def eps_pool_worker(index_row):
    print("Processing %s" % index_row.name)
    try:
        data = load_eps(DATA_DIR, 
                        export_key = index_row['sa_ticker'],
                        primary_ric = index_row['primary_ric'],
                        rics = index_row['rics'])
        
        return data
    except Exception as e:
        print("Exception while processing symbol", index_row['sa_ticker'])
        print('\t', e)
        return {'sa_ticker': index_row['sa_ticker'],
                'rics': index_row['rics'],
                'item': 'EPS',
                'error': e}


def fundamental_pool_worker(index_row, file_suffix, eikon_name):
    print("Processing %s" % index_row.name)
    try:
        data = load_fundamental(DATA_DIR, 
                                file_suffix = file_suffix, eikon_name = eikon_name,
                                export_key = index_row['sa_ticker'],
                                primary_ric = index_row['primary_ric'],
                                rics = index_row['rics'])
        
        return data
    except Exception as e:
        print("Exception while processing symbol", index_row['sa_ticker'])
        print('\t', e)
        return {'sa_ticker': index_row['sa_ticker'],
                'rics': index_row['rics'],
                'item': file_suffix.upper(),
                'error': e}


def events_pool_worker(index_row):
    print("Processing %s" % index_row.name)
    try:
        data = load_events(DATA_DIR, 
                           export_key = index_row['sa_ticker'],
                           primary_ric = index_row['primary_ric'],
                           rics = index_row['rics'])
        
        return data
    except Exception as e:
        print("Exception while processing symbol", index_row['sa_ticker'])
        print('\t', e)
        return {'sa_ticker': index_row['sa_ticker'],
                'rics': index_row['rics'],
                'item': 'EVENT',
                'error': e}



if __name__ == '__main__':
    
    print("Exporting data index ...")
    index = create_data_index(SYMBOL_LIST_FILE)
    
    index.sort_index(inplace = True)
    
    index.to_json(EXPORT_DIR + 'ticker_index.json', orient='records')
    index.to_hdf(EXPORT_DIR + 'ticker_index.hdf', 'index')
    
    #index = index.head(500)
    jobs = [r for n, r in index.iterrows()]
    errors = []
    
    
    print("Exporting adjusted prices for %d tickers ..." % len(index))
    
    with mp.Pool(processes = 16, maxtasksperchild = 10) as pool:
        prices = pool.map(price_pool_worker, jobs, chunksize = 1)
    
    errors += [p for p in prices if type(p) is dict]
    
    prices = pd.concat([p for p in prices if type(p) is pd.DataFrame])
    
    prices.to_hdf(EXPORT_DIR + 'adjusted_prices.hdf', 'prices')
    
    
    print("Exporting market cap for %d tickers ..." % len(index))
    
    with mp.Pool(processes = 16, maxtasksperchild = 10) as pool:
        prices = pool.map(marketcap_pool_worker, jobs, chunksize = 1)
    
    errors += [p for p in prices if type(p) is dict]
    
    prices = pd.concat([p for p in prices if type(p) is pd.DataFrame])
    
    prices.to_hdf(EXPORT_DIR + 'marketcap.hdf', 'marketcap')
    
    
    print("Exporting fundamental ratios for %d tickers ..." % len(index))
    
    with mp.Pool(processes = 16, maxtasksperchild = 10) as pool:
        data = pool.map(fundratio_pool_worker, jobs, chunksize = 1)
    
    errors += [p for p in data if type(p) is dict]
    
    data = pd.concat([p for p in data if type(p) is pd.DataFrame])
    import gc
    gc.collect()
    data.to_hdf(EXPORT_DIR + 'fundratio.hdf', 'fundratio')
    
    
    print("Exporting EPS data for %d tickers ..." % len(index))
    
    with mp.Pool(processes = 16, maxtasksperchild = 100) as pool:
        eps = pool.map(eps_pool_worker, jobs, chunksize = 1)
    
    errors += [p for p in eps if type(p) is dict]
    
    eps = pd.concat([p for p in eps if type(p) is pd.DataFrame])
    
    eps.to_hdf(EXPORT_DIR + 'eps.hdf', 'eps')
    
    
    print("Exporting event data for %d tickers ..." % len(index))
    
    with mp.Pool(processes = 16, maxtasksperchild = 100) as pool:
        events = pool.map(events_pool_worker, jobs, chunksize = 1)
    
    errors += [p for p in events if type(p) is dict]
    
    events = pd.concat([p for p in events if type(p) is pd.DataFrame])
    
    events.to_hdf(EXPORT_DIR + 'events.hdf', 'events')
    
    
    
    for file_suffix, eikon_name in FUNDAMENTALS:
        print("Exporting %s data for %d tickers ..." % (file_suffix, len(index)))
        
        jobs = [(r, file_suffix, eikon_name) for n, r in index.iterrows()]
        
        with mp.Pool(processes = 16, maxtasksperchild = 100) as pool:
            fund = pool.starmap(fundamental_pool_worker, jobs, chunksize = 1)
        
        errors += [p for p in fund if type(p) is dict]
        
        fund = pd.concat([p for p in fund if type(p) is pd.DataFrame] + [pd.DataFrame()])
        
        fund.to_hdf(EXPORT_DIR + file_suffix + '.hdf', file_suffix)
    
    
    
    pd.DataFrame(errors).to_excel(EXPORT_DIR + 'errors.xlsx')

