# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 17:35:01 2023

@author: aq75iwit
"""
import logging
import os
import pandas as pd

from learning_model import run_model, summarize_model_results
from analysis.decile_return import analyze_decile_return
from analysis.feature_importance import analyze_feature_importance
from analysis.return_distribution import analyze_return

os.chdir('c:\\Users\\aq75iwit\\Anaconda3\\envs\\earnings_call_7\\EarningsCall')


def run_job(data, job_id, job, run_folder):
    model_summaries = []
    job_folder = run_folder + '/job-%d/' % job_id
    
    if not os.path.exists(job_folder):
        os.makedirs(job_folder)
    
    try:
        predictions, model_results = run_model(data, job, job_folder)
    except:
        logging.exception("Exception while running job")
        return model_summaries
    
    print('Predictions:',predictions)
    predictions.to_hdf(job_folder + 'predictions.hdf', 'predictions')
    
    pd.Series(model_results).to_hdf(job_folder + 'model_results.hdf', 'results')
    
    summary = summarize_model_results(job, model_results)
    model_summaries.append(pd.concat([pd.Series(job_id, name = 'job_id'), summary], 
                           axis = 0))
    
    analyze_decile_return(model_results, job_folder + 'decile_return')
    analyze_feature_importance(model_results, job_folder + 'feature_importance')
    analyze_return(predictions, job['top_flop_cutoff'], job_folder + 'return')
    save_summaries(model_summaries, job_folder)
    
    return model_summaries


def save_summaries(model_summaries, folder):
    model_summaries = pd.concat(model_summaries, axis = 1).transpose()
    
    writer = pd.ExcelWriter(folder + 'summary.xlsx', engine = 'xlsxwriter')
    model_summaries.to_excel(writer, sheet_name = 'model summary')
    
    #writer.save()
    writer.close()

