# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:34:38 2019

@author: eikon
"""

import os
import pandas as pd


symbol_list_file = 'C:/Users/eikon.WSBI-38/Desktop/EC/symbol_list_excel.xlsx'

out_dir = 'C:/Users/eikon.WSBI-38/Desktop/EC/data/'

DATA_ITEMS = ['PRICE','DIVIDEND','EPS','EVENT']


ticker_list = pd.read_excel(symbol_list_file)
ticker_list = ticker_list[ticker_list['TR.RIC'].isnull()==False]
ticker_list.sort_values('best_ric', inplace = True)
ticker_list.drop_duplicates(subset = ['best_ric'], inplace = True)

result = []
for row, ticker in ticker_list.iterrows():
    for data_type in ['_price.csv', '_eps.csv', '_event.csv', '_div.csv']:
        ric = ticker['best_ric']
        d = {
                'ticker': ric,
                'type': data_type,
                'present': os.path.isfile(out_dir + ric + data_type)*1.0
                }
        result.append(d)

result = pd.DataFrame(result)
result.set_index(['ticker', 'type'], inplace = True)

print(result.unstack(1).mean())
print(result.unstack(1).mean().mean())
