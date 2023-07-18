#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:25:03 2019

@author: mschnaubelt
"""

import numpy as np
import pandas as pd

import string
from rpy2.robjects.packages import importr
import rpy2.robjects as ro

from dask import dataframe as dd

from config import BASE_FEATURES


def add_return_targets(data):
    print("Adding return targets ...")
    
    data['call_return'] = data['Close'] / data['Open'] - 1
    data['cont_return'] = data['Close_next_bday'] / data['Open'] - 1
    data['return'] = data['Close_next_bday'] / data['Open_next_bday'] - 1
    
    data['market_call_return'] = data['Close_market_call_day'] / data['Open_market_call_day'] - 1
    data['market_cont_return'] = data['Close_market_next_bday'] / data['Open_market_call_day'] - 1
    data['market_return'] = data['Close_market_next_bday'] / data['Open_market_next_bday'] - 1
    data['overnight_market_return'] = data['Close_market_call_day'] / data['Open_market_next_bday'] - 1
    
    data['abnormal_call_return'] = data['call_return'] - data['market_call_return']
    data['abnormal_cont_return'] = data['cont_return'] - data['market_cont_return']
    data['abnormal_return'] = data['return'] - data['market_return']
    
    data['pre-drift_return'] = data['Open_next_bday'] / data['Open_-2_bday'] - 1
    data['pre-drift_market'] = data['Open_market_next_bday'] / data['Open_market_-2_bday'] - 1
    data['abnormal_pre-drift'] = data['pre-drift_return'] - data['pre-drift_market']
    
    
    
    for d in [1, 2, 3, 4, 5]:
        data['-%dd_drift' % d] = data['Open_next_bday'] / data['Close_-%d_bday' % d]  - 1
        data['market_-%dd_drift' % d] = data['Open_market_next_bday'] / data['Close_market_-%d_bday' % d] - 1
        data['abnormal_-%dd_drift' % d] = data['-%dd_drift' % d] - data['market_-%dd_drift' % d]
        
        data['ff5_expected_-%dd_drift' % d] = (data.overnight_market_return*data.fama_french_b + 1) \
                            * (data['exp_prod_return_-%d_+0_bday' % (d-1)] + 1) - 1.0
        data['ff5_abnormal_-%dd_drift' % d] = data['-%dd_drift' % d] - data['ff5_expected_-%dd_drift' % d]
        
        data['ff-dec_expected_-%dd_drift' % d] = (data.overnight_market_return + 1) \
                            * (data['decile_exp_prod_return_-%d_+0_bday' % (d-1)] + 1) - 1.0
        data['ff-dec_abnormal_-%dd_drift' % d] = data['-%dd_drift' % d] - data['ff-dec_expected_-%dd_drift' % d]
    
    
    for n in [2, 3, 4, 5, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:
        data['%dd_drift' % n] = data['Close_+%d_bday' % n] / data['Open_next_bday'] - 1
        data['market_%dd_drift' % n] = data['Close_market_+%d_bday' % n] / data['Open_market_next_bday'] - 1
        data['abnormal_%dd_drift' % n] = data['%dd_drift' % n] - data['market_%dd_drift' % n]
        
        data['ff5_expected_%dd_drift' % n] = (data.market_return*data.fama_french_b + 1) \
                            * (data['exp_prod_return_+2_+%d_bday' % n] + 1) - 1.0
        data['ff5_abnormal_%dd_drift' % n] = data['%dd_drift' % n] - data['ff5_expected_%dd_drift' % n]
        
        data['ff-dec_expected_%dd_drift' % n] = (data.market_return + 1) \
                            * (data['decile_exp_prod_return_+2_+%d_bday' % n] + 1) - 1.0
        data['ff-dec_abnormal_%dd_drift' % n] = data['%dd_drift' % n] - data['ff-dec_expected_%dd_drift' % n]
        
        abnormal_intraday_return = data['return'] - data.market_return*data.fama_french_b
        data['ff5_abnormal_prod_%dd_drift' % n] = (abnormal_intraday_return + 1)*\
                                (data['ff5_abnormal_prod_return_+2_+%d_bday' % n] + 1.0)\
                                - 1.0
        data['ff5_abnormal_sum_%dd_drift' % n] = abnormal_intraday_return + data['ff5_abnormal_sum_return_+2_+%d_bday' % n]
    
    
    data['ff-dec_abnormal_1d_drift'] = data['return'] - data['market_return']
    
    
    for n in [-1, -2, -4, -9, -20, -59]:
        data['%dd_pre-drift' % n] = data['Close'] / data['Open_%d_bday' % n] - 1
        data['market_%dd_pre-drift' % n] = data['Close_market_call_day'] / data['Open_market_%d_bday' % n] - 1
        data['abnormal_%dd_pre-drift' % n] = data['%dd_pre-drift' % n] - data['market_%dd_pre-drift' % n]
    
    
    data['random_target'] = np.random.normal(size = len(data)) * 0.1
    
    
    return data


def add_fundamental_features(data, use_lagged_price_scaling = False):
    print("Adding fundamental features ...")
    
    if use_lagged_price_scaling:
        scale_price = data['Price Close Unadj -5bd']
    else:
        scale_price = data['Price Close Unadj']
    
    data['EP_ratio'] = data['EPS Actual'] / scale_price
    
    per = data['Price Close'] / (data['EPS Actual'] + 1E-10)#.apply(lambda x: np.nan if x<=0.0 else x)
    data['pe_ratio'] = per.fillna(0.0)
    
    data['EP_surprise'] = ((data['EPS Actual'] - data['EPS Mean Estimate']) / scale_price).fillna(0.0)
    
    #data['earnings_surprise_smart'] = (data['EPS Actual'] - data['EPS Smart Estimate']) / scale_price
    #data.loc[data.earnings_surprise_smart.isna(), 'earnings_surprise_smart'] = data['earnings_surprise']
    
    data['EP_surprise_last'] = (data['EPS Actual'] - data['EPS Actual Previous Year']) / scale_price
    data.loc[data['EP_surprise_last'].isna(), 'EP_surprise_last'] = data['EP_surprise']
    
    data['EP_surprise_std'] = (data['EPS Estimate Standard Deviation'] / scale_price).fillna(0.0)
    data['EP_surprise_estimates'] = data['EPS Number of Included Estimates'].fillna(1.0)
    
    data['EP_surprise_mean_std'] = (data['EPS Mean Estimate std'] / scale_price).fillna(0.0)
    #data['earnings_surprise_mimax'] = (data['EPS Mean Estimate max'] - data['EPS Mean Estimate min']) / scale_price
    #data['earnings_surprise_grad'] = (data['EPS Mean Estimate'] - data['EPS Mean Estimate first']) / scale_price
    data['EP_surprise_revisions'] = data['EPS Mean Estimate count'].fillna(0.0)
    data['EP_surprise_rev_num'] = data['EPS Mean Estimate count'] / data['EPS Number of Estimates']
    
    
    #eps_yoy_diff = (data['EPS Actual Previous Year'] - data['EPS Actual']) / scale_price
    #eps_qoq_diff = (data['EPS Actual Previous Quarter'] - data['EPS Actual']) / scale_price
    es_est_diff = (data['EPS Actual Previous Year'] - data['EPS Mean Estimate']) / scale_price
    #data['eps_yoy_diff'] = eps_yoy_diff.fillna(0.0)
    #data['eps_qoq_diff'] = eps_qoq_diff.fillna(0.0)
    data['earnings_estimate_diff'] = es_est_diff.fillna(0.0)
    
    #es_lq = (data['EPS Actual Previous Quarter'] - data['EPS Mean Estimate Previous Quarter']) / scale_price
    #data['earnings_surprise_lq'] = es_lq.fillna(0.0)
    
    data['earnings_drift'] = data['EPS Actual Historic Drift'].fillna(0.0) / scale_price
    data['earnings_std'] = data['EPS Actual Historic StD'].fillna(0.0) / scale_price
    data['earnings_ratio_mean'] = data['EPS Actual Historic Mean'].fillna(data['EPS Actual']) / scale_price
    
    
    sp_ratio = 1 / data['Price To Sales Per Share'] / 4.0
    revenue_per_share = data['REVENUE Actual'] / data['Outstanding Shares Adj']
    data['SP_ratio'] = (revenue_per_share / scale_price).fillna(sp_ratio)
    data['SP_surprise'] = (data['REVENUE Actual'] - data['REVENUE Estimate Mean']) / data['Outstanding Shares Adj'] / scale_price
    data['SP_surprise'] = data['SP_surprise'].fillna(0.0)
    data['SP_surprise_std'] = (data['REVENUE Estimate StD'] / data['Outstanding Shares Adj'] / scale_price).fillna(0.0)
    data['SP_surprise_estimates'] = data['REVENUE NIncEstimates'].fillna(0.0)
    
    
    # TODO: define via net income instead?
    margin = (data['NETINC Actual'] - data['OPEX Actual']) / data['REVENUE Actual']
    data['profit_margin'] = margin
    data['GPM_surprise'] = (data['GPM Actual'] - data['GPM Estimate Mean'])
    data['GPM_surprise'] = data['GPM_surprise'].fillna(0)
    
    data['OPEX_surprise'] = (data['OPEX Actual'] - data['OPEX Estimate Mean']) / data['REVENUE Actual']
    data['OPEX_surprise'] = data['OPEX_surprise'].fillna(0)
    
    
    bm = data['BVPS Actual'] / scale_price
    data['BM_ratio'] = bm.fillna(1 / data['Price To Book Value Per Share'])
    data['BM_surprise'] = (data['BVPS Actual'] - data['BVPS Estimate Mean']) / scale_price
    data['BM_surprise'] = data['BM_surprise'].fillna(0.0)
    data['BM_surprise_std'] = (data['BVPS Estimate StD'] / scale_price).fillna(0.0)
    data['BM_surprise_count'] = data['BVPS NIncEstimates'].fillna(0.0)
    
    data['MV_log'] = np.log10(data['Company Market Cap'])
    
    
    dy = data['DIVEST Actual'] / scale_price
    data['DY_ratio'] = dy.fillna(data['TS Rolling Dividend'] / scale_price).fillna(0.0)
    data['DY_surprise'] = (data['DIVEST Actual'] - data['DIVEST Estimate Mean']) / scale_price
    data['DY_surprise'] = data['DY_surprise'].fillna(0.0)
    data['DY_surprise_std'] = (data['DIVEST Estimate StD'] / scale_price).fillna(0.0)
    data['DY_surprise_count'] = data['DIVEST NIncEstimates'].fillna(0.0)
    
    dpr = data['DY_ratio'] / data['EP_ratio']
    data['dividend_payout_ratio'] = dpr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    
    cp = data['CFPS Actual'] / scale_price
    data['CP_ratio'] = cp.fillna(1 / data['Price To Cash Flow Per Share'] / 4).fillna(data['EP_ratio'])
    data['CP_surprise'] = (data['CFPS Actual'] - data['CFPS Estimate Mean']) / scale_price
    data['CP_surprise'] = data['CP_surprise'].fillna(0.0)
    data['CP_surprise_std'] = (data['CFPS Estimate StD'] / scale_price).fillna(0.0)
    data['CP_surprise_count'] = data['CFPS NIncEstimates'].fillna(0.0)
    
    data['ACCRUAL_ratio'] = ((data['EPS Actual'] - data['CFPS Actual']) / scale_price).fillna(0.0)
    #data['cfps_eps_ratio'] = (data['CFPS Actual'] / data['EPS Actual']).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    
    data['fcfps_surprise'] = (data['FCFPS Actual'] - data['FCFPS Estimate Mean']) / scale_price
    data['fcfps_surprise'] = data['fcfps_surprise'].fillna(0.0)
    data['fcfps_ratio'] = data['FCFPS Actual'].fillna(0.0) / scale_price
    data['fcfps_surprise_std'] = (data['FCFPS Estimate StD'] / scale_price)
    data['fcfps_surprise_count'] = data['FCFPS NIncEstimates'].fillna(0)
    
    
    roe_alt = 4*data['NETINC Actual'] / data['SHEQUITY Actual']
    data['ROE_surprise'] = (data['ROE Actual'] - data['ROE Estimate Mean'])
    data['ROE_surprise'] = data['ROE_surprise'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    data['ROE_ratio'] = data['ROE Actual']
    
    
    data['ROA_surprise'] = (data['ROA Actual'] - data['ROA Estimate Mean'])
    data['ROA_surprise'] = data['ROA_surprise'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    data['ROA_ratio'] = data['ROA Actual']
    
    
    
    data['NETDEBT_surprise'] = (data['NETDEBT Actual'] - data['NETDEBT Estimate Mean']) / data['Company Market Cap']
    data['NETDEBT_surprise'] = data['NETDEBT_surprise'].fillna(0)
    
    data['PRETAXINC_surprise'] = (data['PRETAXINC Actual'] - data['PRETAXINC Estimate Mean']) / data['Company Market Cap']
    data['PRETAXINC_surprise'] = data['PRETAXINC_surprise'].fillna(0)
    
    data['NETINC_surprise'] = (data['NETINC Actual'] - data['NETINC Estimate Mean']) / data['Company Market Cap']
    data['NETINC_surprise'] = data['NETINC_surprise'].fillna(0)
    
    data['TOTASSETS_surprise'] = (data['TOTASSETS Actual'] - data['TOTASSETS Estimate Mean']) / data['TOTASSETS Actual']
    data['TOTASSETS_surprise'] = data['TOTASSETS_surprise'].fillna(0)
    
    data['DE_ratio']=data['Total Debt To Enterprise Value'].fillna(0)
    data['EV/OCF_ratio']=data['Enterprise Value To Operating Cash Flow'].fillna(14.5)
    
    day_counts = data.groupby('local_date').local_date.count()
    day_counts.name = 'same_day_call_count'
    data = data.join(day_counts, on = 'local_date')
    data['same_day_call_count_ge_100'] = (data['same_day_call_count'] >= 100)*1.0
    
    data['hour_of_day'] = data.release_datetime.dt.tz_convert('America/New_York').dt.hour.fillna(12)
    data['hour_of_day_half'] = (data['hour_of_day'] >= 12)*1.0
    
    
    data['EP_forward_ratio'] = (data['EPS Forward Estimate'] / scale_price).fillna(data['EP_ratio'])
    data['SP_forward_ratio'] = (data['REVENUE Forward Estimate'] / data['Outstanding Shares Adj'] / scale_price).fillna(data['SP_ratio'])
    data['CP_forward_ratio'] = (data['CFPS Forward Estimate'] / scale_price).fillna(data['CP_ratio'])
    data['BM_forward_ratio'] = (data['BVPS Forward Estimate'] / scale_price).fillna(data['BM_ratio'])
    data['DY_forward_ratio'] = (data['DIVEST Forward Estimate'] / scale_price).fillna(data['DY_ratio'])
    
    
    # calculate trailing valuation ratios from last 4Q values
    data['EP_trailing_ratio'] = (data['EPS Actual Historic Mean'] / scale_price).fillna(data['EP_ratio'])
    data['SP_trailing_ratio'] = (data['REVENUE Actual 4Q Mean'] / data['Outstanding Shares Adj'] / scale_price).fillna(data['SP_ratio'])
    data['CP_trailing_ratio'] = (data['CFPS Actual 4Q Mean'] / scale_price).fillna(data['CP_ratio'])
    data['BM_trailing_ratio'] = (data['BVPS Actual 4Q Mean'] / scale_price).fillna(data['BM_ratio'])
    data['DY_trailing_ratio'] = (data['DIVEST Actual 4Q Mean'] / scale_price).fillna(data['DY_ratio'])
    
    dpr = data['DY_trailing_ratio'] / data['EP_trailing_ratio']
    data['dividend_payout_trailing_ratio'] = dpr.replace([np.inf, -np.inf], np.nan).fillna(data['dividend_payout_ratio'])
    
    
    return data


def add_text_features(data):
    data['log_length'] = np.log10(data.length + 1)
    data['log_length_qanda'] = np.log10(data.length_qanda + 1)
    data['log_length_intro'] = np.log10(data.length_general + 1)
    
    return data


def add_rolling_features(data):
    # calculate rolling z-scored features
    data = data.sort_values('final_datetime')
    
    for f in BASE_FEATURES:
        r_mean = data.rolling(1500, min_periods = 100)[f].mean()
        r_std = data.rolling(1500, min_periods = 100)[f].std()
        
        data[f + ':roll-z'] = ((data[f] - r_mean) / r_std).fillna(0.0)
    
    return data


def add_sentiment_features(data):
    print("Adding sentiment features ...")
    
    def get_sentiment(docs):
        sentimentAnalysis = importr('SentimentAnalysis')
        
        d = docs.apply(lambda x: ' '.join(x) if type(x) is list else x)
        d = d.apply(lambda s: ''.join([x for x in s if x in string.printable]))
        
        sen = sentimentAnalysis.analyzeSentiment(ro.StrVector(d))
        
        r = {}
        for i, name in enumerate(sen.names):
            r[name] = np.array(sen[i])
        
        return pd.DataFrame(r, index = docs.index)
    
    print("\tfrom general part ...")
    S_gen = dd.from_pandas(data.general, npartitions = 100).\
               map_partitions(get_sentiment).\
               compute(scheduler = 'processes', num_workers = 6)
    
    print("\tfrom Q&A part ...")    
    S_qanda = dd.from_pandas(data.qanda, npartitions = 100).\
               map_partitions(get_sentiment).\
               compute(scheduler = 'processes', num_workers = 6)
    
    S_gen.fillna(0, inplace = True)
    S_qanda.fillna(0, inplace = True)
    
    data = pd.concat([data, S_gen.add_prefix('RSenti_general_')], axis = 1)
    data = pd.concat([data, S_qanda.add_prefix('RSenti_qanda_')], axis = 1)
    
    
    return data

