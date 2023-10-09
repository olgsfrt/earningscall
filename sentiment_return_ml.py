#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:58:19 2019

@author: mschnaubelt
"""
import os

os.getcwd()

import logging
import datetime
import pandas as pd
import importlib

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import RandomForestRegressor

from util.validator import PhysicalTimeForwardValidation
#from util.prepare_data import prepare_data, clean_data

from job_runner_23 import run_job, save_summaries

from config import RUNS_FOLDER, BASE_FEATURES
import pandas as pd

#os.chdir('c:/Users/aq75iwit/Anaconda3/envs/earnings_call_5/EarningsCall/')

#run_job=0
#save_summaries=0

"""
#'features_2d_ale'#'feature_set_time_dependence'#'feature_importances_ale'#'final_regression'
JOB_CONFIG_FILE = 'final_backtest'
RUN_BACKTESTS = False





data = prepare_data(
        #call_file = '/mnt/data/earnings_calls/data_changes_21_11_19.json.bak_22122019'
        use_lagged_price_scaling = True
        )
data = clean_data(data)

data = data.sort_values('final_datetime')
data.reset_index(inplace = True)
"""

#data=pd.read_pickle('D:/01_Diss_Data/00_Data_Final/merged_data_17_05_2023.pickle')
#data=pd.read_pickle('D:/01_Diss_Data/00_Data_Final/testsample.pkl')
#data=pd.read_pickle('D:/01_Diss_Data/00_Data_Final/data_years_2007_2011.pkl')
data=pd.read_pickle('D:/01_Diss_Data/00_Data_Final/data_years_2007_2008.pkl')
#='D:/01_Diss_Data/00_Data_Final/data_years_2007_2011.pkl'
#data=0

data['final_datetime'] = pd.to_datetime(data['final_datetime'], unit='ms')
#data['final_datetime'] = data['final_datetime'].dt.strftime('%Y-%m-%d')


JOB_CONFIG_FILE = 'final'
data = data.sort_values('final_datetime')
data.reset_index(drop=True, inplace=True)
runn = '%s' % JOB_CONFIG_FILE

#data['final_datetime'] = pd.to_datetime(data['final_datetime'])

#data.to_json('/mnt/data/earnings_calls/data.json', orient = 'records')



#model = BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=100)
"""
# =============================================================================
model = RandomForestRegressor(n_estimators = 5000, max_depth = 20, #min_samples_split = 0.01,
                               #class_weight = "balanced_subsample", 
                               random_state = 0, n_jobs = -1)
# =============================================================================
"""
# import ANN model from models/ann.py

from sklearn.linear_model import LassoCV, LinearRegression, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, KBinsDiscretizer, FunctionTransformer

#model = LassoCV(max_iter = 2000, random_state = 0,
#                cv = 5, n_alphas = 100, n_jobs = -1)
#model = Pipeline([('scaler', KBinsDiscretizer(n_bins = 5)), ('LR', model)])


#from sklearn.linear_model import LassoCV, LinearRegression, ElasticNetCV
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import QuantileTransformer, KBinsDiscretizer, FunctionTransformer

#model = LassoCV(max_iter = 2000, random_state = 0,
#                cv = 5, n_alphas = 100, n_jobs = -1)
#model = Pipeline([('scaler', KBinsDiscretizer(n_bins = 5)), ('LR', model)])
"""
job = {
       'train_subset': 'SP1500',
       'model': model,
       'train_target': 'ff5_abnormal_20d_drift',
       'return_target': 'ff5_abnormal_20d_drift',
       'features': BASE_FEATURES,
       'top_flop_cutoff': 0.1,
       'validator': PhysicalTimeForwardValidation('2013-01-01', pd.Timedelta(365, 'D'), 
                                                   1500, 'final_datetime'),
       'rolling_window_size': 1500,
       'calculate_permutation_feature_importances': False,
       'calculate_partial_dependence': False,
       'calculate_single_ale': False,
       'calculate_dual_ale': []
      }

job = {
       'train_subset': 'SP500TR',
       'model': model,
       'train_target': 'ff5_abnormal_20d_drift',
       'return_target': 'ff5_abnormal_20d_drift',
       'features': BASE_FEATURES,
       'top_flop_cutoff': 0.1,
       'validator': PhysicalTimeForwardValidation('2007-05-01', pd.Timedelta(20, 'D'), 
                                                   50, 'final_datetime'),
       'rolling_window_size': 100,
       'calculate_permutation_feature_importances': True,
       'calculate_partial_dependence': False,
       'calculate_single_ale': False,
       'calculate_dual_ale': []
      }
"""


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
chandler = logger.handlers[0]
cformatter = logging.Formatter('%(levelname)s - %(message)s')
chandler.setFormatter(cformatter)

config_mod = importlib.import_module('job_definitions.%s' % JOB_CONFIG_FILE)

model_jobs = config_mod.generate_model_jobs()

logging.info("Loaded %d model jobs", len(model_jobs))

ts = datetime.datetime.now().replace(microsecond = 0)\
            .isoformat().replace(':', '_')

#runn = "ml_run" 
run_folder = RUNS_FOLDER + '/%s-%s/' % (runn if runn is not None else 'run', ts)

logging.info("Writing results to run folder %s", run_folder)

model_summaries = []

#print(data.shape)
#print(data.head())

for job_id, job in enumerate(model_jobs):
    logging.info("Running job %d of %d", job_id, len(model_jobs))
    
    model_summary = run_job(data, job_id, job, run_folder) # Adjusted function call
    
    model_summaries += model_summary

S = model_summaries[0]

save_summaries(model_summaries, run_folder)  # Adjusted function call

""""
job = {
       'train_subset': 'SP1500',
       'model': model,
       'train_target': 'ff-dec_abnormal_20d_drift',
       'return_target': 'ff-dec_abnormal_20d_drift',
       'features': BASE_FEATURES,
       'top_flop_cutoff': 0.1,
       'validator': PhysicalTimeForwardValidation('2013-01-01', pd.Timedelta(12, 'M'), 
                                                   1500, 'final_datetime'),
       'rolling_window_size': 1500,
       'calculate_permutation_feature_importances': False,
       'calculate_partial_dependence': False,
       'calculate_single_ale': False,
       'calculate_dual_ale': []#[['EP_ratio', 'EP_forward_ratio']]#[list(x) for x in itertools.combinations(['CP_ratio', 'EP_ratio', 
                                                           #'MV_log', 'EP_surprise', 
                                                           #'SP_ratio', 'BM_ratio', 
                                                           #'EP_surprise_std', 
                                                           #'general_NegativityLM', 
                                                           #'EP_surprise_mean_std', 
                                                          # 'CP_surprise', 'SP_surprise'], 2)]
      }





for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
chandler = logger.handlers[0]
cformatter = logging.Formatter('%(levelname)s - %(message)s')
chandler.setFormatter(cformatter)


config_mod = importlib.import_module('job_definitions.%s' % JOB_CONFIG_FILE)

model_jobs = config_mod.generate_model_jobs()#[job]
bt_jobs = config_mod.generate_backtest_jobs()

logging.info("Loaded %d model jobs", len(model_jobs))
logging.info("Loaded %d backtest jobs", len(bt_jobs))



ts = datetime.datetime.now().replace(microsecond = 0)\
            .isoformat().replace(':', '_')
run_folder = RUNS_FOLDER + '/%s-%s/' % (runn if runn is not None else 'run', ts)

logging.info("Wrinting results to run folder %s", run_folder)


model_summaries = []
backtest_summaries = []

for job_id, job in enumerate(model_jobs):
    logging.info("Running job %d of %d", job_id, len(model_jobs))
    
    model_summary, backtest_summary = run_job(data, job_id, job, bt_jobs, run_folder,
                                              run_bt = RUN_BACKTESTS)
    
    model_summaries += model_summary
    backtest_summaries += backtest_summary

S = model_summaries[0]


save_summaries(model_summaries, backtest_summaries, run_folder)

"""

exit()

"""
import numpy as np
if __name__ == "__main__":
    # Generate Sample Data for testing
    np.random.seed(42)
    n_samples = 5000
    date_rng = pd.date_range(start='2010-01-01', periods=n_samples, freq='D')
    X = pd.DataFrame(date_rng, columns=['final_datetime'])
    X['data'] = np.random.randint(0, 100, size=(n_samples))

    val = PhysicalTimeForwardValidation('2012-01-01', pd.Timedelta(365, 'D'), 
                                        1000, 'final_datetime')
    
    for i, (train_index, test_index, val_index) in enumerate(val.split(X)):
        
        train_data = X.loc[train_index]
        test_data = X.loc[test_index]
        val_data = X.loc[val_index]
        
        print("Split #%d" % i)
        print("Train data of length %d from %s to %s" % (len(train_data), 
                            train_data.final_datetime.min(), train_data.final_datetime.max()))
        print("Validation data of length %d from %s to %s" % (len(val_data), 
                            val_data.final_datetime.min(), val_data.final_datetime.max()))
        print("Test data of length %d from %s to %s" % (len(test_data), 
                            test_data.final_datetime.min(), test_data.final_datetime.max()))
        print("-----")

"""