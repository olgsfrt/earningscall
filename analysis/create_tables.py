#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:52:52 2019

@author: mschnaubelt
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import FEATURE_SETS_ORDER, FINAL_RUN_FOLDERS
from analysis_helper import extract_job_info, read_run

from return_distribution import analyze_return_by_tr_sector, extended_describe


def extract_model_comparison_column(R):
    C = {}
    
    C['01Count'] = int(R['count'])
    C['021Mean'] = R['mean'] * 100
    C['022Mean (long)'] = R['deciles_returns']['mean'].iloc[5:].mean() * 100
    C['023Mean (short)'] = -R['deciles_returns']['mean'].iloc[0:5].mean() * 100
    C['03t-statistic'] = R['t']
    
    C['04Minimum'] = R['min'] * 100
    C['05First quart.'] = R['25%'] * 100
    C['06Median'] = R['50%'] * 100
    C['07Third quart.'] = R['75%'] * 100
    C['08Maximum'] = R['max'] * 100
    
    C['09Std. dev.'] = R['std'] * 100
    
    #C['05Median p-value'] = R['median_pvalue']
    if 'test_mse' in R:
        C['10MSE'] = R['test_mse'] * 100
    if 'test_r2' in R:
        C['11R2'] = R['test_r2'] * 100
    C['12Dir. acc.'] = R['test_bacc'] * 100
    
    C['51Count top-flop'] = int(R['top_flop_count'])
    C['521Mean top-flop'] = R['top_flop_mean'] * 100
    C['522Mean (long)'] = R['deciles_returns']['mean'].tail(1).mean() * 100
    C['523Mean (short)'] = -R['deciles_returns']['mean'].head(1).mean() * 100
    C['53t-statistic'] = R['top_flop_t']
    
    C['54Minimum top-flop'] = R['top_flop_min'] * 100
    C['55First quart. top-flop'] = R['top_flop_25%'] * 100
    C['56Median top-flop'] = R['top_flop_50%'] * 100
    C['57Third quart. top-flop'] = R['top_flop_75%'] * 100
    C['58Maximum top-flop'] = R['top_flop_max'] * 100
    
    C['59Std. dev. top-flop'] = R['top_flop_std'] * 100
    
    #C['15Median p-value'] = R['top_flop_median_pvalue']
    if 'top_flop_mse' in R:
        C['60MSE top-flop'] = R['top_flop_mse'] * 100
    if 'top_flop_r2' in R:
        C['61R2 top-flop'] = R['top_flop_r2'] * 100
    C['62Dir. acc. top-flop'] = R['top_flop_bacc'] * 100
    
    return pd.Series(C)



def process_run(run_folder):
    
    results = read_run(run_folder)
    results = results.apply(extract_model_comparison_column, axis = 1)
    
    ex_results = results.stack()
    ind_n = list(ex_results.index.names)
    ind_n[-1] = 'kpi'
    ex_results.index.names = ind_n
    
    uex_results = ex_results.unstack(level = [1, 0])
    
    SR = uex_results[[c for c in uex_results.columns.values if 'EP-ratio' not in c[1]]]
    SR = SR.loc[(['VR', 'DIS+FE+VR', 'FE+POL+UIQ+VR+54'], )]
    SR.sort_index(axis = 1, level = 0, inplace = True)
    
    def fmter(x):
        if pd.isna(x):
            return '--'
        else:
            return '%.3f' % x if abs(x) < 1000 else '%d' % x
    
    SR.to_excel(run_folder + '/model_comparison.xlsx')
    
    SRL = SR.loc[:, ([5, 20, 60], ['0 ES-EP-surprise', '1 LR', '1 LR-B10', '8 RF-D20-E5000'])]
    L = SRL.to_latex(float_format = fmter, multicolumn = True)
    L = re.sub('[ ]+', ' ', L)
    with open(run_folder + '/model_comparison.tex', "w") as f:
        print(L, file = f)
    
    
    FR_rows = ~ex_results.reset_index().kpi.str.contains('Count')
    FR_rows &= ~ex_results.reset_index().features.str.contains('TFR')
    #dis = ex_results.reset_index().features.str.contains('DIS', regex=False)
    #pol = ex_results.reset_index().features.str.contains('POL', regex=False)
    #FR_rows &= ~(dis & pol)
    
    FR = ex_results.loc[('8 RF-D20-E5000', ), FR_rows].reset_index(level=0, drop=True).unstack(level = 1)
    feature_sets = FR.columns.str.split('+')
    feature_set_row_names = []
    for i, fs in enumerate(FEATURE_SETS_ORDER):
        rn = '%d-%s' % (i, fs)
        FR = FR.append(pd.Series(feature_sets, name = (0, rn), index = FR.columns).apply(lambda x: fs in x))
        feature_set_row_names.append(rn)
    
    FR.sort_index(inplace = True)
    FR.sort_values([(0, fs) for fs in feature_set_row_names[::-1]], axis = 1, inplace = True)
    
    L = FR.to_latex(float_format = fmter, multicolumn = True)
    L = re.sub('[ ]+', ' ', L)
    with open(run_folder + '/feature_set_comparison.tex', "w") as f:
        print(L, file = f)
    
    FR.to_excel(run_folder + '/feature_set_comparison.xlsx')
    
    

def process_run_index_segment(run_folder):
    
    results = read_run(run_folder)
    
    SR = []
    for i, segment in enumerate(['SP500TR', 'SP400TR', 'SP600TR']):
        s = results.apply(lambda r: extract_model_comparison_column(r[segment + '_results']), axis = 1)
        s = pd.concat([s], keys = [str(i) + segment], names = ['mkt_index'])
        SR.append(s)
    SR = pd.concat(SR)
    
    SR = SR.loc[(slice(None), '8 RF-D20-E5000', [5, 20, 60], 'DIS+FE+FR+POL+54'), ]
    SR.index = SR.index.droplevel([1, 3])
    
    SR['100Count fraction'] = SR['51Count top-flop'] / SR['01Count']
    SR['101Return quotient'] = SR['52Mean top-flop'] / SR['02Mean']
    
    SR = SR.sort_index(level=1).transpose().swaplevel(0, 1, axis = 1)
    
    
    def fmter(x):
        if pd.isna(x):
            return '--'
        else:
            return '%.3f' % x if abs(x) < 1000 else '%d' % x
    
    L = SR.to_latex(float_format = fmter, multicolumn = True)
    L = re.sub('[ ]+', ' ', L)
    with open(run_folder + '/index_segment_comparison.tex', "w") as f:
        print(L, file = f)
    
    SR.to_excel(run_folder + '/index_segment_comparison.xlsx')


def process_run_industry_sector(run_folder, alpha = 0.1):
    
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
        if info['features'] != 'DIS+FE+FR+POL+54':
            continue
        
        period = info['period']
        
        if not os.path.isfile(run_folder + '%s/predictions.hdf' % job):
            continue
        
        preds = pd.read_hdf(run_folder + '%s/predictions.hdf' % job)
        preds['is_top_flop'] = (preds.pred_rank < alpha) | (preds.pred_rank > 1 - alpha)
        
        sector_results = analyze_return_by_tr_sector(preds)
        sector_results = pd.concat([sector_results], keys = [period], names = ['period'])
        
        SR.append(sector_results)
    
    SR = pd.concat(SR)
    SR = SR.loc[([-5, 5, 60], ['all', 'top-flop'], ), ]
    SR = SR.unstack(level = 1).swaplevel(0, 1, axis = 1)
    
    SR = SR.loc[:, [(d, c) for d in ['top-flop', 'all'] for c in ['count', 'mean', 't']]]
    SR['Fraction'] = SR.loc[:, ('top-flop', 'count')] / SR.loc[:, ('all', 'count')]
    
    SR.loc[:, [('top-flop', 'count'), ('all', 'count')]] = SR.loc[:, [('top-flop', 'count'), ('all', 'count')]].astype(int)
    
    def fmter(x):
        if pd.isna(x):
            return '--'
        elif type(x) is int:
            return '%d' % x
        else:
            return '%.3f' % x if abs(x) < 1000 else '%d' % x
    
    L = SR.to_latex(float_format = fmter, multicolumn = True)
    L = re.sub('[ ]+', ' ', L)
    with open(run_folder + '/industry_sector_comparison.tex', "w") as f:
        print(L, file = f)
    
    SR.to_excel(run_folder + '/industry_sector_comparison.xlsx')


def process_run_call_count(run_folder, alpha = 0.1):
    
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
        if info['features'] != 'DIS+FE+FR+POL+54':
            continue
        
        period = info['period']
        
        if not os.path.isfile(run_folder + '%s/predictions.hdf' % job):
            continue
        
        preds = pd.read_hdf(run_folder + '%s/predictions.hdf' % job)
        preds['is_top_flop'] = (preds.pred_rank < alpha) | (preds.pred_rank > 1 - alpha)
        preds = preds.join(preds.groupby('local_date').local_date.count(),
                           on = 'local_date', rsuffix = '_count')
        
        bins = [(1, 10), (11, 20), (21, 50), (51, 100), (101, 200)]
        bins = [(1, 23), (24, 55), (56, 85), (86, 109), (110, 1000)]
        for i, (left, right) in enumerate(bins):
            sel_preds = preds[preds.is_top_flop & preds.local_date_count.between(left, right)]
            r = extended_describe(sel_preds.signed_target_return)
            r = pd.concat([r], keys = ['%d %d to %d' % (i, left, right)], 
                          names = ['bin'])
            r = pd.concat([r], keys = [period], 
                          names = ['period'])
            
            SR.append(r)
    
    SR = pd.concat(SR)
    
    SR = SR.loc[([-5, 5, 60], slice(None), slice(None)), ]
    SR = SR.unstack(level = 2)
    
    
    def fmter(x):
        if pd.isna(x):
            return '--'
        elif type(x) is int:
            return '%d' % x
        else:
            return '%.3f' % x if abs(x) < 1000 else '%d' % x
    
    L = SR.to_latex(float_format = fmter, multicolumn = True)
    L = re.sub('[ ]+', ' ', L)
    with open(run_folder + '/call_count_comparison.tex', "w") as f:
        print(L, file = f)
    
    SR.to_excel(run_folder + '/call_count_comparison.xlsx')


def process_run_event_quarter(run_folder, alpha = 0.1):
    
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
        if info['features'] != 'DIS+FE+FR+POL+54':
            continue
        
        period = info['period']
        
        if not os.path.isfile(run_folder + '%s/predictions.hdf' % job):
            continue
        
        preds = pd.read_hdf(run_folder + '%s/predictions.hdf' % job)
        preds['is_top_flop'] = (preds.pred_rank < alpha) | (preds.pred_rank > 1 - alpha)
        preds['ld_quarter'] = preds.final_datetime.dt.to_period("Q").astype(str)
        
        sel_preds = preds[preds.is_top_flop]
        r = sel_preds.groupby('ld_quarter').signed_target_return.apply(extended_describe)
        
        r = pd.concat([r], keys = [period], names = ['period'])
            
        SR.append(r)
    
    SR = pd.concat(SR).unstack(level = 2)
    
    
    fig = plt.figure(figsize = (12, 4))
    ax = fig.subplots(1, 1)
    
    ax.axhline(0, color = 'lightgray')
    
    x = SR.index.levels[1].values
    periods = [5, 20, 60]
    width = (1.0 - 0.2) / len(periods)
    x_offset = - len(periods)/2 * width + width/2
    for i, p in enumerate(periods):
        sel = SR.loc[(p, list(x)), ]
        
        ax.bar(np.arange(len(x)) + x_offset + i*width, sel['mean'] * 100, 
               width, label = '%d days' % p)
    
    x_labels = [c if i%2 == 0 else '' for i, c in enumerate(x)]
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x_labels)
    
    ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
    ax.set_ylabel('Mean abnormal return', fontsize = 13)
    ax.set_xlim(0 + x_offset - width, len(x) - 1 - x_offset + width)
    
    ax.legend(fontsize = 12)
    
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.0, wspace = 0.0,
                        left = 0.07, right = 0.98,
                        bottom = 0.07, top = 0.98)
    
    fig.savefig(run_folder + 'return_by_quarter.pdf')
    plt.close()
    



if __name__ == '__main__':
    for run_folder in FINAL_RUN_FOLDERS:
        process_run(run_folder)
        process_run_index_segment(run_folder)
        process_run_industry_sector(run_folder)
        process_run_call_count(run_folder)
        process_run_event_quarter(run_folder)
