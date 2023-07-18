#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:24:06 2019

@author: mschnaubelt
"""

import numpy as np

from util.join_data import get_joined_call_data
from util.add_features import add_return_targets, add_fundamental_features, \
                                add_text_features, add_rolling_features

from config import CALL_FILE


def prepare_data(call_file = CALL_FILE, use_us_subset = True, use_lagged_price_scaling = True):
    data = get_joined_call_data(call_file)
    
    data.reset_index(inplace = True)
    
    data = add_return_targets(data)
    data = add_fundamental_features(data, use_lagged_price_scaling = use_lagged_price_scaling)
    data = add_text_features(data)
    data = add_rolling_features(data)
    
    if use_us_subset:
        data = data[data.mkt_index.isin(['NONE:US','SP400TR','SP500TR','SP600TR'])]
    
    return data


def clean_data(data, clean_return_columns = True, 
               clean_surprise_columns = False, clean_fund = True):
    print("Cleaning data ...")
    
    data.replace([np.inf, -np.inf], np.nan, inplace = True)
    
    if clean_return_columns:
        data.dropna(subset = ['abnormal_return', 'abnormal_5d_drift', 'ff5_abnormal_5d_drift', 
                              'abnormal_20d_drift', 'abnormal_60d_drift'], 
                    inplace = True)
    
    if clean_surprise_columns:
        data.dropna(subset = ['EP_surprise'], 
                    inplace = True)
    
    if clean_fund:
        data.dropna(subset = ['MV_log', 'EP_ratio', 'BM_ratio'], 
                    inplace = True)
    
    # TODO: drop negative DY, SP values?
    
    return data



def get_index_counts(data):
    
    const_quarter_counts = data[~data.mkt_index.isna()].groupby(['fiscal_period','mkt_index'])\
                        .ticker_symbol.count().unstack(1)
    
    cdata = data[~data.mkt_index.isna()]
    const_counts = cdata.groupby([cdata.final_datetime.dt.year, 'mkt_index'])\
                        .ticker_symbol.count().unstack(1)
    
    print("Post-2010 call count:", const_counts[const_counts.index>=2010].sum(axis=1).sum())
    
    const_counts.to_excel('SP_constituent_counts.xlsx')
    
    const_counts['total'] = const_counts.sum(axis = 1)
    const_counts['S&P 400 coverage'] = const_counts['SP400TR'] / (4*400) *100
    const_counts['S&P 500 coverage'] = const_counts['SP500TR'] / (4*500) *100
    const_counts['S&P 600 coverage'] = const_counts['SP600TR'] / (4*600) *100
    sp1500 = const_counts['SP400TR'] + const_counts['SP500TR'] + const_counts['SP600TR']
    const_counts['S&P 1500 coverage'] = sp1500 / (4*1500) *100
    
    const_counts = const_counts.transpose()
    const_counts['sum'] = const_counts.sum(axis=1)
    
    return const_counts, const_quarter_counts

