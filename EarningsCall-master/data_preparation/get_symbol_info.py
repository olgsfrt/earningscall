# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:20:09 2019

@author: eikon
"""

import time
import re
import pandas as pd
import eikon as ek
from difflib import SequenceMatcher

ticker_list_file = 'C:/Users/eikon.WSBI-38/Desktop/EC/reuters_id_6.xlsx'
out_dir = 'C:/Users/eikon.WSBI-38/Desktop/EC/'

ek.set_app_key('ac2e022cefad4af595f4517611492f7c9d1b918a')


def get_ticker_list(ticker_list_file, debug = False):
    ticker_list = pd.read_excel(ticker_list_file)
    ticker_list = ticker_list.astype(str)
    if debug:
        ticker_list = ticker_list.head(100)
    
    ord_tickers = ticker_list[(ticker_list.best_ric.str.contains('.', regex=False) == True) 
        & (ticker_list.best_ric.str.contains('\.O($|\^)', regex=True) == False)].copy()
    
    ord_tickers['best_ric'] = ord_tickers.best_ric.str.replace('\.[a-zA-Z]+', '.O')
    
    #ticker_list = pd.concat([ticker_list, ord_tickers], axis = 0)
    #ticker_list.drop_duplicates(subset = 'best_ric', inplace = True)
    
    return ticker_list


ticker_list = get_ticker_list(ticker_list_file, False)


### compare Seeking Alpha Names with Eikon name from best_ric ###

regex = re.compile('[^a-zA-Z]')
ticker_list['sa_name_clean'] = ticker_list['sa_name'].str.lower().str.replace('\([a-z:]*\)', '')\
           .str.replace('.', '').str.replace(',', '').str.replace('corporation', '')\
            .str.replace('inc', '').str.replace('ltd', '').str.replace('corp', '')


ticker_list['name_diff'] = ticker_list.apply(lambda r: SequenceMatcher(None, r['sa_name_clean'], 
           r['tr_best_ric_common_name'].lower().replace('.', '').replace('inc', '').replace('ltd', '').replace('corp', '')).ratio(), 
           axis = 1)
E = ticker_list[['sa_ticker','sa_name','best_ric','tr_best_ric_common_name','manually_checked?','in_manual?','name_diff']]

E.to_excel('company_name_compare.xlsx')


### also include primary quote RIC to try to retrieve all available data ###
excel_tickers = ticker_list[['sa_ticker', 'best_ric']]

if True:
    symbol_list = pd.read_excel('C:/Users/eikon.WSBI-38/Desktop/EC/symbol_list_excel.xlsx')
    symbol_list = symbol_list[symbol_list['TR.RIC'].isnull()==False]
    symbol_list.sort_values('best_ric', inplace = True)
    
    add_excel_tickers = symbol_list.loc[symbol_list['TR.PrimaryRIC'] != symbol_list['TR.RIC'], 
                                        ['sa_ticker', 'TR.PrimaryRIC']]
    add_excel_tickers.columns = ['sa_ticker', 'best_ric']
    
    excel_tickers = pd.concat([excel_tickers, add_excel_tickers])

excel_tickers.sort_values('best_ric', inplace = True)
excel_tickers.to_excel('excel_tickers.xlsx')






symbol_info = []

for row, ticker in ticker_list.iterrows():
    s = None
    
    while s is None:
        print("Requesting ticker %d of %d" % (row, len(ticker_list)))
        try:
            s = ek.get_symbology(str(ticker['best_ric']), bestMatch = True).iloc[0]
        except ek.EikonError as e:
            err = e
            if e.code == 429:
                print("\tWaiting for API for 60 seconds...")
                time.sleep(60)
    
    symbol_info.append(pd.DataFrame(pd.concat([ticker, s])).transpose())
    
    time.sleep(0.25)

symbol_info = pd.concat(symbol_info)

symbol_info.to_hdf(out_dir + 'symbol_info.hdf', 'data')
