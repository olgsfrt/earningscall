#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:57:30 2020

@author: mschnaubelt
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import FINAL_TIME_DEP_RUN, FINAL_ALE_RUN, BASE_FEATURES
from config import RATIO_FEATURES, FORECAST_ERROR_FEATURES, DISPERSION_FEATURES
from create_tables import extract_model_comparison_column
from analysis_helper import extract_job_info, read_run

from return_distribution import analyze_return_by_tr_sector, extended_describe



def process_run_index_segment_features(run_folder):
    
    results = read_run(run_folder)
    
    SR = []
    for i, segment in enumerate(['SP500TR', 'SP400TR', 'SP600TR']):
        s = results.apply(lambda r: extract_model_comparison_column(r[segment + '_results']), axis = 1)
        s = pd.concat([s], keys = [str(i) + segment], names = ['mkt_index'])
        SR.append(s)
    SR = pd.concat(SR)
    
    SR = SR[['52Mean top-flop', '02Mean']]
    
    SR = SR.loc[(slice(None), '8 RF-D20-E5000', slice(None), slice(None)), ]
    SR = SR.unstack(level = 3)
    
    SR = SR.sort_index(level = 2)
    
    SR.to_excel(run_folder + '/return_by_segment_and_feature_set.xlsx')



def process_run_industry_features(run_folder, alpha = 0.1):
    
    jobs = [f for f in os.listdir(run_folder) if f.startswith('job')]
    jobs.sort(key = lambda s: int(s.replace('job-', '')))
    
    SR = []
    
    for job in jobs:
        if not os.path.isfile(run_folder + '%s/model_results.hdf' % job):
            continue
        
        result = pd.read_hdf(run_folder + '%s/model_results.hdf' % job)
        info = extract_job_info(result)
        
        if info['model'] != '8 RF-D20-E5000':
            continue
        
        period = info['period']
        if period not in [-5, 5, 20, 60]:
            continue
        
        feature_set = info['features']
        
        if not os.path.isfile(run_folder + '%s/predictions.hdf' % job):
            continue
        
        preds = pd.read_hdf(run_folder + '%s/predictions.hdf' % job)
        preds['is_top_flop'] = (preds.pred_rank < alpha) | (preds.pred_rank > 1 - alpha)
        
        sector_results = analyze_return_by_tr_sector(preds)
        sector_results = pd.concat([sector_results], keys = [period], names = ['period'])
        sector_results = pd.concat([sector_results], keys = [feature_set], names = ['featues'])
        
        SR.append(sector_results)
    
    SR = pd.concat(SR)['mean']
    
    SR = SR.loc[(slice(None), [-5, 5, 60], ['all', 'top-flop'], ), ]
    SR = SR.unstack(level = 2).unstack(level = 0)
    
    SR.to_excel(run_folder + '/return_by_industry_and_feature_set.xlsx')



def process_run_book_market_features(run_folder, alpha = 0.1, 
                                     class_feature = 'BM_ratio'):
    
    jobs = [f for f in os.listdir(run_folder) if f.startswith('job')]
    jobs.sort(key = lambda s: int(s.replace('job-', '')))
    
    SR = []
    
    for job in jobs:
        if not os.path.isfile(run_folder + '%s/model_results.hdf' % job):
            continue
        
        result = pd.read_hdf(run_folder + '%s/model_results.hdf' % job)
        info = extract_job_info(result)
        
        if info['model'] != '8 RF-D20-E5000':
            continue
        
        period = info['period']
        if period not in [-5, 5, 20, 60]:
            continue
        
        feature_set = info['features']
        if feature_set != 'FE+POL+UIQ+VR+54':
            continue
        
        if not os.path.isfile(run_folder + '%s/predictions.hdf' % job):
            continue
        
        preds = pd.read_hdf(run_folder + '%s/predictions.hdf' % job)
        preds['is_top_flop'] = (preds.pred_rank < alpha) | (preds.pred_rank > 1 - alpha)
        preds['bin'] = pd.qcut(preds[class_feature], [0.0, 0.2, 0.8, 1.0], duplicates = 'drop')
        
        
        all_result = preds.groupby('bin')['signed_target_return'].agg(['mean', 'count'])
        all_result['mean'] *= 1E4
        all_result.columns = pd.MultiIndex.from_product([['all'], all_result.columns])
        
        s_preds = preds[preds['is_top_flop']]
        top_flop_result = s_preds.groupby('bin')['signed_target_return'].agg(['mean', 'count'])
        top_flop_result['mean'] *= 1E4
        top_flop_result.columns = pd.MultiIndex.from_product([['top_flop'], top_flop_result.columns])
        top_flop_result.index = all_result.index
        
        R = pd.concat([all_result, top_flop_result], axis = 1)
        
        R[('', 'count ratio')] = R[('top_flop', 'count')] / R[('all', 'count')]
        
        R.index = R.index.astype(str)
        R = pd.concat([R], keys = [period], names = ['period'])
        R = pd.concat([R], keys = [feature_set], names = ['feature_set'])
        
        SR.append(R)
    
    SR = pd.concat(SR)
    
    SR_mean = SR.loc[:, (slice(None), 'mean')].unstack(level = 0)
    SR_mean.columns = SR_mean.columns.droplevel(1)
    
    SR.to_excel(run_folder + '/return_by_%s_and_feature_set_20percent.xlsx' % class_feature)



def process_run_variable_quantiles(run_folder, 
                                   feature_name = 'EP_surprise_std', 
                                   n_quantiles = 5):
    
    jobs = [f for f in os.listdir(run_folder) if f.startswith('job')]
    jobs.sort(key = lambda s: int(s.replace('job-', '')))
    
    SR = []
    
    for job in jobs:
        if not os.path.isfile(run_folder + '%s/model_results.hdf' % job):
            continue
        
        result = pd.read_hdf(run_folder + '%s/model_results.hdf' % job)
        info = extract_job_info(result)
        
        if info['model'] != '8 RF-D20-E5000':
            continue
        
        period = info['period']
        
        preds = pd.read_hdf(run_folder + '%s/predictions.hdf' % job)
        
        preds['is_top_flop'] = (preds.pred_rank < 0.1) | (preds.pred_rank > 1 - 0.1)
        preds['bin'] = pd.qcut(preds[feature_name], n_quantiles, duplicates = 'drop')
        
        all_result = preds.groupby('bin')['signed_target_return'].agg(['mean', 'count'])
        all_result['mean'] *= 1E4
        all_result.columns = pd.MultiIndex.from_product([['all'], all_result.columns])
        
        s_preds = preds[preds['is_top_flop']]
        top_flop_result = s_preds.groupby('bin')['signed_target_return'].agg(['mean', 'count'])
        top_flop_result['mean'] *= 1E4
        top_flop_result.columns = pd.MultiIndex.from_product([['top_flop'], top_flop_result.columns])
        top_flop_result.index = all_result.index
        
        R = pd.concat([all_result, top_flop_result], axis = 1)
        
        R[('', 'count ratio')] = R[('top_flop', 'count')] / R[('all', 'count')]
        
        R.index = R.index.astype(str)
        R = pd.concat([R], keys = [period], names = ['period'])
        
        SR.append(R)
    
    SR = pd.concat(SR)
    
    SR.to_excel('%s/return_by_%s.xlsx' % (run_folder, feature_name))



def evaluate_mean_feature_values(run_folder, alpha = 0.1):
    
    jobs = [f for f in os.listdir(run_folder) if f.startswith('job')]
    jobs.sort(key = lambda s: int(s.replace('job-', '')))
    
    SR = {}
    
    for job in jobs:
        if not os.path.isfile(run_folder + '%s/model_results.hdf' % job):
            continue
        
        result = pd.read_hdf(run_folder + '%s/model_results.hdf' % job)
        info = extract_job_info(result)
        
        if info['model'] != '8 RF-D20-E5000':
            continue
        
        period = info['period']
        if period not in [-5, 5, 20, 60]:
            continue
        
        feature_set = info['features']
        if feature_set != 'FE+POL+UIQ+VR+54':
            continue
        
        if not os.path.isfile(run_folder + '%s/predictions.hdf' % job):
            continue
        
        preds = pd.read_hdf(run_folder + '%s/predictions.hdf' % job)
        
        preds = preds.join(data.set_index(['ticker_symbol', 'fiscal_period', 'local_date'])[BASE_FEATURES],
                           on = ['ticker_symbol', 'fiscal_period', 'local_date'], rsuffix = '_')
        
        preds['is_top_flop'] = (preds.pred_rank < alpha) | (preds.pred_rank > 1 - alpha)
        preds['long_short'] = 'none'
        preds.loc[preds.pred_rank < alpha, 'long_short'] = 'short'
        preds.loc[preds.pred_rank > 1 - alpha, 'long_short'] = 'long'
        
        R = preds.groupby('long_short')[RATIO_FEATURES+FORECAST_ERROR_FEATURES+
                          DISPERSION_FEATURES].mean().transpose()
        
        R['all'] = preds[RATIO_FEATURES+FORECAST_ERROR_FEATURES+
                         DISPERSION_FEATURES].mean()
        
        R['long_all_ratio'] = R['long'] / R['all']
        R['short_all_ratio'] = R['short'] / R['all']
        
        SR[period] = R
    
    SR = pd.concat(SR, axis = 1)
    
    SR.to_excel(run_folder + 'mean_feature_values.xlsx')




if __name__ == '__main__':
    process_run_industry_features(run_folder = FINAL_TIME_DEP_RUN)
    process_run_index_segment_features(run_folder = FINAL_TIME_DEP_RUN)
    
    for f in BASE_FEATURES:
        process_run_variable_quantiles(run_folder = FINAL_ALE_RUN, feature_name = f)



