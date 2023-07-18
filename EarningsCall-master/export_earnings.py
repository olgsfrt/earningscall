#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:05:08 2019

@author: mschnaubelt
"""

import os
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
import multiprocessing as mp

from util import read_call_data


in_file = '/mnt/data/earnings_calls/meta_v17.json'
data_folder = '/home/mschnaubelt/tmp/hf-data/'
out_folder = '/mnt/data/earnings_calls/'


print('Reading call data')
data = read_call_data(in_file)


#data = data.head(50)


def get_prices(report_data, symbol = None, index = 0):
    if symbol is None:
        symbol = report_data.name
    
    print("Processing symbol %s (#%d)" % (symbol, index))
    
    report_data = report_data.dropna(subset=['datetime'])
    
    price_matrix = []
    price_matrix.append(pd.Series(symbol, index = report_data.index, 
                                  name = "ticker_symbol").reset_index(drop = True))
    price_matrix.append(report_data.datetime.reset_index(drop = True))
    
    price_matrix.append(pd.Series(True, index = report_data.index, 
                                  name = "in_sp500").reset_index(drop = True))
    
    
    filename = '%s/table_%s.csv' % (data_folder, symbol.lower())
    
    if not os.path.isfile(filename):
        print("\tSymbol not found!")
        price_matrix = pd.concat(price_matrix, axis = 1)
        price_matrix['in_sp500'] = False
        return price_matrix
    
    print("\tGot %d earnings reports" % len(report_data))
    
    ts_data = pd.read_csv(filename,
                          names = ['date', 'time', 
                                   'open', 'high', 'low', 'close', 'volume',
                                   'splits', 'earnings', 'dividends'],
                          dtype = {'date': str, 'time': int})
    
    
    dt = ts_data.apply(lambda r: "%s %0.4d" % (r.date, r.time), axis = 1)
    dt = pd.to_datetime(dt, format='%Y%m%d %H%M').dt.tz_localize('America/New_York')
    ts_data.set_index(dt, inplace = True)
    
    ts_data = ts_data.resample('1min').bfill(limit=5)
    
    def get_dt(report_dt):
        report_dt = report_dt.tz_convert('America/New_York')
        
        times = []
        times.append(('mkt_open', report_dt.replace(hour = 9, minute = 30)))
        times.append(('mkt_close', report_dt.replace(hour = 16, minute = 0)))
        
        times += [('call_+%d' % d, report_dt + pd.Timedelta(d, unit = 'm')) 
                    for d in [0, 60, 120, 24*60, 7*24*60]]
        
        
        times.append(('next_bday_open', (report_dt + BDay(1)).replace(hour = 9, minute = 30)))
        times.append(('next_bday_close', (report_dt + BDay(1)).replace(hour = 16, minute = 0)))
        
        return pd.DataFrame([dict(times)])
    
    ts = pd.concat([get_dt(v) for v in report_data.datetime], axis = 0)
    ts.reset_index(drop = True, inplace = True)
    
    
    for c, times in ts.iteritems():
        times = times.dt.tz_convert('America/New_York')
        
        
        price_matrix.append(times)
        
        try:
            ps = ts_data.open.loc[np.asarray(times, dtype = object)]
        except:
            continue
        
        ps.index = times.index
        ps.name = 'open_' + times.name
        
        price_matrix.append(ps)
    
    price_matrix = pd.concat(price_matrix, axis = 1)
    
    return price_matrix



if __name__ == '__main__':
    
    def pool_worker(eport_data, symbol, index):
        try:
            return get_prices(eport_data, symbol, index)
        except Exception as e:
            print("Exception while processing symbol", symbol)
            print(e)
    
    print("Earnings reports contain %d unique ticker symbols" % len(data.ticker_symbol.unique()))

    groups = [(d, n, i) for i, (n, d) in enumerate(data.groupby('ticker_symbol'))]
    
    
    with mp.Pool(processes = 16, maxtasksperchild = 1) as pool:
        result = pool.starmap(pool_worker, groups, chunksize = 1)
    
    result = pd.concat(result)
    
    print("Finished, getting SPY data")
    
    
    spy_result = get_prices(data, symbol = 'SPY')
    
    spy_result = spy_result[['datetime'] + [c for c in spy_result.columns if 'open_' in c]]
    
    result = result.merge(spy_result, how = 'left', 
                          left_on = ['datetime'], right_on = ['datetime'], 
                          suffixes = ['', '_spy'])
    
    result.sort_index(inplace = True)
    result.set_index(['ticker_symbol', 'datetime'], inplace = True)
    
    result.to_hdf(out_folder + 'earnings_prices.hdf', 'result')
    result.to_csv(out_folder + 'earnings_prices.csv')
    
    result.reset_index(inplace = True)
    
    result.to_json(out_folder + 'earnings_prices.json', 
                   orient = 'records', date_format = 'iso')
    
