#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:59:43 2019

@author: mschnaubelt
"""

import os

import datetime
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from util.validator import PhysicalTimeForwardValidation
from util.prepare_data import prepare_data, clean_data

from config import RUNS_FOLDER
from learning_model import run_model, summarize_model_results


data = prepare_data(
        call_file = '/mnt/data/earnings_calls/con_dict_01_08_19.json',
        add_sentiment = True
        )

data = clean_data(data)


data = data.sort_values('final_datetime')
data.reset_index(inplace = True)

data = data[data['local_date'] < '2013-01-01']



TUNED_MODEL = RandomForestClassifier(n_estimators = 1000, #max_depth = 10, 
                                     class_weight = "balanced_subsample", 
                                     random_state = 0, n_jobs = -1)

#TUNED_PARAM = 'min_samples_split'
#PARAM_VALUES = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
#PARAM_VALUES = [10, 25, 50, 100, 150, 200, 250, 350, 500, 750, 1000]

TUNED_PARAM = 'max_depth'
PARAM_VALUES = [3, 4, 5, 6, 8, 10, 15]


job = {
       'train_subset': 'SP1500',
       'model': TUNED_MODEL,
       'train_target': 'abnormal_5d_drift',
       'return_target': 'abnormal_5d_drift',
       'features': ['earnings_surprise', 
                    'earnings_surprise_mean_std', 
                    'earnings_surprise_std', 
                    'earnings_surprise_revisions', 
                    'earnings_surprise_estimates', 
                    'earnings_ratio', 
                    'pays_dividend', 
                    'revenue_surprise', 
                    'revenue_surprise_estimates', 
                    'log_length', 
                    'nr_analysts', 
                    'general_PositivityLM', 'general_NegativityLM', 
                    'qanda_PositivityLM', 'qanda_NegativityLM', 
                    ],
        'top_flop_cutoff': 0.1,
        'validator': PhysicalTimeForwardValidation('2011-01-01', pd.Timedelta(3, 'M'), 
                                                   1500, 'final_datetime'),
        'rolling_window_size': 1500,
        'calculate_permutation_feature_importances': False
       }


ts = datetime.datetime.now().replace(microsecond = 0).isoformat().replace(':', '_')
run_folder = RUNS_FOLDER + '/tuning-run-%s/' % ts

if not os.path.exists(run_folder):
    os.makedirs(run_folder)


model_summaries = []

for i, param_value in enumerate(PARAM_VALUES):
    
    job['model'].set_params(**{TUNED_PARAM: param_value})
    
    predictions, model_results = run_model(data, job)
    
    summary = summarize_model_results(job, model_results)
    
    model_summaries.append(summary)


model_summaries = pd.concat(model_summaries, axis = 1)

model_summaries.to_excel(run_folder + 'tuning_summary.xlsx', 
                         sheet_name = 'tuning summary')

