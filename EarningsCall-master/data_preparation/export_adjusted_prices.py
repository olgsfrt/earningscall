#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:04:45 2019

@author: mschnaubelt
"""

import os
import pandas as pd
import numpy as np



def load_raw_dividend_data(data_dir, export_key = 'AAPL',
                           primary_ric = 'AAPL', rics = ['AAPL.OQ', 'AAPL.O']):
    
    files = ['%s/%s_div.csv' % (data_dir, s) for s in rics]
    
    data = [pd.read_csv(f) for f in files if os.path.isfile(f)]
    data = [d for d in data if 'Dividend Ex Date' in d.columns]
    
    if len(data) == 0:
        return None
    
    div_data = pd.concat(data)
    
    div_data['Dividend Ex Date'] = pd.to_datetime(div_data['Dividend Ex Date'], utc = True)
    div_data['Dividend Pay Date'] = pd.to_datetime(div_data['Dividend Pay Date'], utc = True)
    
    div_data.dropna(subset = ['Dividend Ex Date', 'Currency'], inplace = True)
    
    div_data.rename(index = str, columns={"Currency": "Div Currency"}, inplace = True)

    
    div_data['instrument_key'] = export_key
    
    div_data.drop_duplicates(subset = ['Dividend Ex Date', 'Div Currency'],
                             inplace = True)
    
    return div_data[['instrument_key', 'Dividend Ex Date', 'Dividend Pay Date', 'Div Currency', 
                     'Tax Status', 'Net Dividend Amount', 'Gross Dividend Amount', 
                     'Adjusted Net Dividend Amount', 'Adjusted Gross Dividend Amount']]



def load_raw_price_data(data_dir, export_key = 'AAPL',
                        primary_ric = 'AAPL.O', rics = ['AAPL.OQ', 'AAPL.O']):
    
    files = ['%s/%s_price.csv' % (data_dir, s) for s in rics]
    
    data = [pd.read_csv(f) for f in files if os.path.isfile(f)]
    data = [d for d in data if 'Price Close' in  d.columns]
    
    if len(data) == 0:
        return None
    
    data = pd.concat(data)
    data.dropna(subset = ['Date', 'Price Close'], inplace = True)
    
    data['Date'] = pd.to_datetime(data['Date'], utc = True)
    
    data['is_primary'] = (data.Instrument == primary_ric)*1.0
    data['instrument_key'] = export_key
    
    data.sort_values('is_primary', ascending = False, inplace = True)
    
    data.drop_duplicates(subset = ['Date'],
                         inplace = True)
    
    data.sort_values('Date', ascending = True, inplace = True)
    
    return data[['instrument_key', 'Date', 'Currency',
                 'Price Open', 'Price Close', 'Price High', 'Price Low', 
                 'Volume']]



def load_adjusted_prices(data_dir, export_key, primary_ric, rics):
    div_data = load_raw_dividend_data(data_dir, export_key, primary_ric, rics)
    price_data = load_raw_price_data(data_dir, export_key, primary_ric, rics)
    
    if price_data is None or len(price_data) == 0:
        raise Exception('No price data for ticker %s!' % export_key)
    
    if div_data is None:
        print("Warning: No dividend data for %s" % export_key)
        div_data = pd.DataFrame(columns = ['instrument_key', 'Dividend Ex Date', 
                                           'Dividend Pay Date', 'Div Currency', 
                                           'Tax Status', 'Net Dividend Amount', 
                                           'Gross Dividend Amount', 
                                           'Adjusted Net Dividend Amount', 
                                           'Adjusted Gross Dividend Amount'])
    
    min_price_date = price_data['Date'].min()
    
    div_data = div_data[div_data['Dividend Ex Date'] >= min_price_date]
    
    
    div_data['div'] = np.where(div_data['Tax Status'] != 'Gross dividends', 
            div_data['Adjusted Net Dividend Amount'],
            div_data['Adjusted Gross Dividend Amount'])
    
    div_data.set_index(['instrument_key', 'Dividend Ex Date'], inplace = True)
    price_data.set_index(['instrument_key', 'Date'], inplace = True, drop = False)
    
    
    price_data = pd.concat([price_data, div_data[['div', 'Div Currency']]], axis = 1)
    price_data['Currency'] = price_data.Currency.fillna(axis = 0, method = 'ffill', limit = 1)
    
    adj_price = price_data['Price Close'].shift(1).fillna(axis = 0, method = 'ffill', limit = 5)
    
    ccy_check = price_data.apply(lambda r: np.nan if pd.isnull(r['Div Currency']) 
                                    else str(r['Div Currency']).upper() == str(r['Currency']).upper(), 
                                 axis = 1)
    if not ccy_check.all():
        raise Exception('Dividend and stock price currency do not always match for ticker %s!' %
                        export_key)
    
    adj = (1 - price_data['div'] / adj_price).fillna(1.0)
    price_data['Factor'] = 1 / adj.cumprod()
    price_data['Factor'] = price_data['Factor'] / price_data['Factor'].iloc[-1]
    
    price_data['Open'] = price_data['Price Open'] * price_data['Factor']
    price_data['Close'] = price_data['Price Close'] * price_data['Factor']
    price_data['Low'] = price_data['Price Low'] * price_data['Factor']
    price_data['High'] = price_data['Price High'] * price_data['Factor']
    
    return price_data.reset_index(drop=True)


if __name__ == '__main__':
    price_data = load_adjusted_prices(DATA_DIR, export_key = 'AAPL',
                         primary_ric = 'AAPL.O', rics = ['AAPL.OQ', 'AAPL.O'])
    price_data['Close Return'] = price_data['Close'].pct_change()
    
    
    # compare to TR Apple data
    tr_data = pd.read_csv('/home/mschnaubelt/Downloads/reuters/xtsSaniSP500.csv')[['Date', 'X992816']]
    
    tr_data = tr_data.set_index(pd.to_datetime(tr_data['Date'], utc = True))
    tr_data['ret'] = tr_data.X992816.pct_change()
    
    
    comp = pd.concat([tr_data['ret'], price_data['Close Return'].set_index('level_1')], 
                      axis = 1, join = 'inner')
    pct = (comp['ret'] - comp['Close Return']) 
    
    (comp[['ret', 'Close Return']].tail(2000)+1).cumprod()

