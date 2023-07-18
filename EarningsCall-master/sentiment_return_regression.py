#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:58:19 2019

@author: mschnaubelt
"""

import pandas as pd
import statsmodels.api as sm

from util.prepare_data import prepare_data, clean_data
from config import TARGETS, FEATURES


data = prepare_data()
data = clean_data(data)

#data = data.sample(5000)


C = data[TARGETS + FEATURES].corr(method = 'spearman')


jobs = [
        {
            'subset': 'SP1500', 'target': 'abnormal_cont_return',
            'features': ['earnings_surprise', 'log_length', 'nr_analysts', 'nr_executives',
                         'pays_dividend', 'general_SentimentLM', 'general_RatioUncertaintyLM', 
                         'qanda_SentimentLM', 'qanda_RatioUncertaintyLM']
        },
        {
            'subset': 'SP1500', 'target': 'abnormal_return',
            'features': ['earnings_surprise', 'log_length', 'nr_analysts', 'nr_executives',
                         'pays_dividend', 'general_SentimentLM', 'general_RatioUncertaintyLM', 
                         'qanda_SentimentLM', 'qanda_RatioUncertaintyLM',
                         'abnormal_call_return']
        },
        {
            'subset': 'SP1500', 'target': 'abnormal_5d_drift',
            'features': ['earnings_surprise', 'log_length', 'nr_analysts', 'nr_executives',
                         'pays_dividend', 'general_SentimentLM', 'general_RatioUncertaintyLM', 
                         'qanda_SentimentLM', 'qanda_RatioUncertaintyLM',
                         'abnormal_call_return']
        },
        {
            'subset': 'SP1500', 'target': 'abnormal_5d_drift',
            'features': ['earnings_surprise', 'log_length', 'nr_analysts', 'nr_executives',
                         'pays_dividend', 'general_PositivityLM', 'general_NegativityLM',
                         'general_RatioUncertaintyLM', 
                         'abnormal_call_return']
        },
        {
            'subset': 'SP1500', 'target': 'abnormal_5d_drift',
            'features': ['earnings_surprise', 'log_length', 'nr_analysts', 'nr_executives',
                         'pays_dividend', 'general_PositivityLM', 'general_NegativityLM',
                         'general_RatioUncertaintyLM', 
                         'abnormal_call_return'] + ['%d_b' % (i, ) for i in range(30)]
        },
        {
            'subset': 'SP1500', 'target': 'abnormal_5d_drift',
            'features': ['earnings_surprise', 'log_length', 'nr_analysts', 'nr_executives',
                         'pays_dividend', 'whole_general_neg', 'whole_general_pos', 
                         'whole_general_unc',
                         'abnormal_call_return']
        },
        {
            'subset': 'SP500TR', 'target': 'abnormal_5d_drift',
            'features': ['earnings_surprise', 'log_length', 'nr_analysts', 'nr_executives',
                         'pays_dividend', 'general_SentimentLM', 'general_RatioUncertaintyLM', 
                         'qanda_SentimentLM', 'qanda_RatioUncertaintyLM',
                         'abnormal_call_return']
        },
        {
            'subset': 'SP600TR', 'target': 'abnormal_5d_drift',
            'features': ['earnings_surprise', 'log_length', 'nr_analysts', 'nr_executives',
                         'pays_dividend', 'general_SentimentLM', 'general_RatioUncertaintyLM', 
                         'qanda_SentimentLM', 'qanda_RatioUncertaintyLM',
                         'abnormal_call_return']
        },
        {
            'subset': None, 'target': 'abnormal_5d_drift',
            'features': ['earnings_surprise', 'log_length', 'nr_analysts', 'nr_executives',
                         'pays_dividend', 'general_SentimentLM', 'general_RatioUncertaintyLM', 
                         'qanda_SentimentLM', 'qanda_RatioUncertaintyLM',
                         'abnormal_call_return']
        },
        {
            'subset': 'SP1500', 'target': 'abnormal_return',
            'features': ['earnings_surprise', 'nr_analysts', 'nr_executives',
                         'pays_dividend', 'whole_general_neg', 'whole_general_pos',
                         'whole_general_unc',
                         'abnormal_call_return'] + ['%d_%s' % (i, d) for i in range(30) for d in ['pos', 'neg']]
        },
        {
            'subset': 'SP1500', 'target': 'abnormal_5d_drift',
            'features': ['earnings_surprise', 'nr_analysts', 'nr_executives',
                         'pays_dividend', 'whole_general_neg', 'whole_general_pos',
                         'whole_general_unc',
                         'abnormal_call_return'] + ['%d_%s' % (i, d) for i in range(30) for d in ['pos', 'neg']]
        },
        {
            'subset': 'SP1500', 'target': 'ff5_abnormal_5d_drift',
            'features': ['MV_log', 'BM_ratio', 'EP_ratio', 'SP_ratio', 
                         #'CP_ratio', 'ACCRUAL_ratio',
                         'DY_ratio', 'dividend_payout_ratio',
                         'BM_surprise', 'EP_surprise', 'SP_surprise', 
                         'DY_surprise', #'CP_surprise',
                         'EP_surprise_mean_std', 'EP_surprise_std', 
                         'EP_surprise_revisions', 'EP_surprise_estimates', 
                         'SP_surprise_std', 'SP_surprise_estimates', 
                         'DY_surprise_std', 'log_length', 'nr_analysts',
                         'general_PositivityLM', 'general_NegativityLM', 
                         'qanda_PositivityLM', 'qanda_NegativityLM']
        }
        ]



for job in jobs:
    index_names = ['SP500TR', 'SP400TR', 'SP600TR'] if job['subset'] == 'SP1500' \
                    else [job['subset']]
    if job['subset'] is not None:
        reg_data = data.loc[data.mkt_index.isin(index_names)]
    else:
        reg_data = data
    
    regr_columns = job['features']
    
    if not pd.Series(regr_columns).isin(data.columns).all():
        continue
    
    inputs = (reg_data[regr_columns] - reg_data[regr_columns].mean()) / reg_data[regr_columns].std()

    x = sm.add_constant(reg_data[regr_columns], 
                        prepend = False)
    
    mod = sm.OLS(reg_data[job['target']], x)
    res = mod.fit(cov_type = 'HAC', cov_kwds = {'maxlags': 10})
    
    print("\n\n\n*** Using data subset from indices %s ***\n" % index_names)
    print(res.summary2())



