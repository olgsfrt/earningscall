#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:03:02 2019

@author: mschnaubelt
"""

import logging
import os
import pandas as pd

from learning_model import run_model, summarize_model_results
from trading_simulation import run_backtest

from analysis.decile_return import analyze_decile_return
from analysis.feature_importance import analyze_feature_importance
from analysis.return_distribution import analyze_return


def run_job(data, job_id, job, backtest_jobs, run_folder, run_bt = True):
    model_summaries = []
    backtest_summaries = []
    
    job_folder = run_folder + '/job-%d/' % job_id
    
    if not os.path.exists(job_folder):
        os.makedirs(job_folder)
    
    try:
        predictions, model_results = run_model(data, job, job_folder)
    except:
        logging.exception("Exception while running job")
        return model_summaries, backtest_summaries
    
    predictions.to_hdf(job_folder + 'predictions.hdf', 'predictions')
    pd.Series(model_results).to_hdf(job_folder + 'model_results.hdf', 'results')
    
    summary = summarize_model_results(job, model_results)
    model_summaries.append(pd.concat([pd.Series(job_id, name = 'job_id'), summary], 
                           axis = 0))
    
    
    analyze_decile_return(model_results, job_folder + 'decile_return')
    analyze_feature_importance(model_results, job_folder + 'feature_importance')
    analyze_return(predictions, job['top_flop_cutoff'], job_folder + 'return')
    
    if not run_bt:
        return model_summaries, backtest_summaries
    
    for backtest_id, backtest_config in enumerate(backtest_jobs):
        
        name = backtest_config['name'] if 'name' in backtest_config else ''
        
        backtest_folder = job_folder + '/backtest-%d-%s/' % (backtest_id, name)
        
        if not os.path.exists(backtest_folder):
            os.makedirs(backtest_folder)
        
        bt_result, pf_returns = run_backtest(predictions, backtest_config, backtest_folder)
        
        bt_result['model_run'] = job_id
        
        backtest_summaries.append(pd.Series(bt_result))
    
    save_summaries(model_summaries, backtest_summaries, job_folder)
    
    return model_summaries, backtest_summaries



def save_summaries(model_summaries, backtest_summaries, folder):
    model_summaries = pd.concat(model_summaries, axis = 1).transpose()
    
    if len(backtest_summaries) > 0:
        backtest_summaries = pd.concat(backtest_summaries, axis = 1).transpose()
        
        backtest_cols = ['model_run', 'name', 'strategy', 'strategy_args', 
                         'Annual return', 'Mean daily return', 'Mean daily t-statistic (NW)',
                         'Sharpe ratio', 'Max leverage']
        backtest_cols += [c for c in backtest_summaries.columns if c not in backtest_cols]
    else:
        backtest_summaries = pd.DataFrame()
    
    writer = pd.ExcelWriter(folder + 'summary.xlsx', engine = 'xlsxwriter')
    
    model_summaries.to_excel(writer, sheet_name = 'model summary')
    
    if len(backtest_summaries) > 0:
        backtest_summaries[backtest_cols].to_excel(writer, sheet_name = 'backtest summary')
    
    writer.save()


