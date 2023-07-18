#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:57:07 2019

@author: mschnaubelt
"""

import logging
import datetime
import pandas as pd

from models.random_model import MonkeyModel

from util.validator import PhysicalTimeForwardValidation
from util.prepare_data import prepare_data, clean_data

from job_runner import run_job, save_summaries

from config import RUNS_FOLDER, FEATURES



data = prepare_data()
data = clean_data(data)


data = data.sort_values('final_datetime')
data.reset_index(inplace = True)



job = {
       'train_subset': 'SP1500',
       'model': MonkeyModel(),
       'train_target': 'ff5_abnormal_5d_drift',
       'return_target': 'ff5_abnormal_5d_drift',
       'features': FEATURES,
        'top_flop_cutoff': 0.1,
        'validator': PhysicalTimeForwardValidation('2013-01-01', pd.Timedelta(12, 'M'), 
                                                   1500, 'final_datetime'),
        'rolling_window_size': 1500,
        'calculate_permutation_feature_importances': False,
        'calculate_partial_dependence': False
       }



model_jobs = [job for i in range(1000)]
bt_jobs = []


ts = datetime.datetime.now().replace(microsecond = 0)\
            .isoformat().replace(':', '_')
run_folder = RUNS_FOLDER + '/%s-%s/' % ('run-monkey', ts)

logging.info("Wrinting results to run folder %s", run_folder)


model_summaries = []
backtest_summaries = []

for job_id, job in enumerate(model_jobs):
    logging.info("Running job %d of %d", job_id, len(model_jobs))
    
    model_summary, backtest_summary = run_job(data, job_id, job, bt_jobs, run_folder,
                                              run_bt = False)
    
    model_summaries += model_summary
    backtest_summaries += backtest_summary


S = model_summaries[0]

model_summaries = pd.concat(model_summaries, axis = 1)
model_summaries = model_summaries.transpose()

stat = model_summaries['test.return.mean'].astype(float).describe()

save_summaries(model_summaries, backtest_summaries, run_folder)
stat.to_csv(run_folder + 'monkey_stats.csv')
