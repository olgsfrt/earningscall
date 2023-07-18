#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:58:19 2019

@author: mschnaubelt
"""

import pickle
import os
import logging
import datetime
import pandas as pd
import numpy as np
import importlib

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import keras

from util.validator import PhysicalTimeForwardValidation
from util.prepare_data import prepare_data, clean_data

from learning_model import run_model, summarize_model_results

from trading_simulation import run_backtest

from config import RUNS_FOLDER


JOB_CONFIG_FILE = 'final'
RUN_BACKTESTS = True


data = prepare_data(
        #call_file = '/mnt/data/earnings_calls/con_dict_01_08_19.json',
        #call_file = '/mnt/data/earnings_calls/topic_sentiments_intro_qanda_consolidated_27_08_19.json',
        #call_file = '/mnt/data/earnings_calls/con_dict_similarity_clusters_ohe_09_09_19.json',
        call_file = '/home/mschnaubelt/Downloads/con_dict_detailed_qanda_and_sentiments_10_09_19.json',
        add_sentiment = True
        )

data = clean_data(data)




tmp_file = '/mnt/data/earnings_calls/tmp/all.pickle'

if False:
    with open(tmp_file, "wb") as f:
        pickle.dump(data, f)
    
    with open(tmp_file, 'rb') as f:
        data = pickle.load(f)


data = data.sort_values('final_datetime')
data.reset_index(inplace = True)




def create_model():
    from keras.regularizers import l2
    
    classifier = keras.Sequential()
    
    classifier.add(keras.layers.Dense(15, input_dim = 15, activation = 'relu'))
    #classifier.add(keras.layers.Dropout(0.01))
    #classifier.add(keras.layers.Dense(10, activation = 'tanh'))
    #classifier.add(keras.layers.Dropout(0.1))
    #classifier.add(keras.layers.Dense(5, activation = 'tanh'))
    #classifier.add(keras.layers.Dropout(0.1))
    classifier.add(keras.layers.Dense(4, activation = 'relu', 
                                      activity_regularizer=l2(1E-3)
                                      ))
    #classifier.add(keras.layers.Dropout(0.01))
    classifier.add(keras.layers.Dense(2, activation = 'softmax', 
                                      #activity_regularizer=l2(1E-5)
                                      ))
    
    opt = keras.optimizers.Nadam()
    
    classifier.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    
    return classifier

earl = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, min_delta = 0)

from keras.wrappers.scikit_learn import KerasClassifier

    
logdir = RUNS_FOLDER + "/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model = KerasClassifier(build_fn = create_model, batch_size = 128, epochs = 200, 
                        verbose = 1, #callbacks = [tensorboard_callback]
                        )

model = Pipeline([('scaler', StandardScaler()), ('ANN', model)])




job = {
       'train_subset': 'SP1500',
       'model': model,
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
                    #'revenue_surprise_std', 
                    'revenue_surprise_estimates', 
                    #'same_day_call_count', 
                    #'hour_of_day_half', 
                    'log_length', 
                    'nr_analysts', #'nr_executives',
                    'general_PositivityLM', 'general_NegativityLM', 
                    'qanda_PositivityLM', 'qanda_NegativityLM', 
                    
                    #'general_SentimentLM', 'qanda_SentimentLM',
                    #'general_PositivityHE', 'general_NegativityHE', 
                    #'qanda_PositivityHE', 'qanda_NegativityHE', 
                    #'call_return',
                    #'-1d_pre-drift', '-2d_pre-drift', 
                    #'-4d_pre-drift',
                    #'-9d_pre-drift', '-20d_pre-drift',
                    #'es_drift_9', 'es_drift_5'
                    #'healthcare', 'industrial-goods', 'technology', 'financial', 'utilities', 
                    #'consumer-goods', 'services', 'basic-materials','conglomerates'
                    ], #+ ['%d_b'%c for c in [10, 25, 29]],
        'top_flop_cutoff': 0.1,
        'validator': PhysicalTimeForwardValidation('2013-01-01', pd.Timedelta(12, 'M'), 
                                                   1500, 'final_datetime'),
        'rolling_window_size': 1500,
        'calculate_permutation_feature_importances': False
       }



def run_job(job_id, job, backtest_jobs, run_folder, run_bt = True):
    model_summaries = []
    backtest_summaries = []
    
    job_folder = run_folder + '/job-%d/' % job_id
    
    if not os.path.exists(job_folder):
        os.makedirs(job_folder)
    
    predictions, model_results = run_model(data, job)
    
    predictions.to_hdf(job_folder + 'predictions.hdf', 'predictions')
    pd.Series(model_results).to_hdf(job_folder + 'model_results.hdf', 'results')
    
    summary = summarize_model_results(job, model_results)
    model_summaries.append(pd.concat([pd.Series(job_id, name = 'job_id'), summary], 
                           axis = 0))
    
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
    backtest_summaries = pd.concat(backtest_summaries, axis = 1).transpose()
    
    backtest_cols = ['model_run', 'name', 'strategy', 'strategy_args', 
                     'Annual return', 'Mean daily return', 'Mean daily t-statistic (NW)',
                     'Sharpe ratio', 'Max leverage']
    backtest_cols += [c for c in backtest_summaries.columns if c not in backtest_cols]
    
    writer = pd.ExcelWriter(folder + 'summary.xlsx', engine = 'xlsxwriter')
    
    model_summaries.to_excel(writer, sheet_name = 'model summary')
    backtest_summaries[backtest_cols].to_excel(writer, sheet_name = 'backtest summary')
    
    writer.save()


# TODO: rank market segments separately?


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
chandler = logger.handlers[0]
cformatter = logging.Formatter('%(levelname)s - %(message)s')
chandler.setFormatter(cformatter)


config_mod = importlib.import_module('job_definitions.%s' % JOB_CONFIG_FILE)

model_jobs = [job]#config_mod.generate_model_jobs()#[job]
bt_jobs = config_mod.generate_backtest_jobs()

logging.info("Loaded %d model jobs", len(model_jobs))
logging.info("Loaded %d backtest jobs", len(bt_jobs))


runn = 'ann-run'
ts = datetime.datetime.now().replace(microsecond = 0)\
            .isoformat().replace(':', '_')
run_folder = RUNS_FOLDER + '/%s-%s/' % (runn if runn is not None else 'run', ts)

logging.info("Wrinting results to run folder %s", run_folder)


model_summaries = []
backtest_summaries = []

for job_id, job in enumerate(model_jobs):
    logging.info("Running job %d of %d", job_id, len(model_jobs))
    
    model_summary, backtest_summary = run_job(job_id, job, bt_jobs, run_folder,
                                              run_bt = RUN_BACKTESTS)
    
    model_summaries += model_summary
    backtest_summaries += backtest_summary

S = model_summaries[0]


save_summaries(model_summaries, backtest_summaries, run_folder)

