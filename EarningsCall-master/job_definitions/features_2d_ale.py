#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 08:13:22 2019

@author: mschnaubelt
"""

import itertools
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from util.validator import PhysicalTimeForwardValidation
from config import BASE_FEATURES



TARGETS = ['ff-dec_abnormal_-5d_drift', 'ff-dec_abnormal_5d_drift', 
           'ff-dec_abnormal_20d_drift', 'ff-dec_abnormal_60d_drift']
MIN_RANK = 0.1



GENERAL_MODEL_CONFIG = {
        'train_subset': 'SP1500',
        'top_flop_cutoff': MIN_RANK,
        'validator': PhysicalTimeForwardValidation('2013-01-01', pd.Timedelta(12, 'M'), 
                                                   1500, 'final_datetime'),
        'rolling_window_size': 1500,
        'calculate_permutation_feature_importances': False,
        'calculate_partial_dependence': False,
        'calculate_single_ale': False,
        'calculate_dual_ale': [list(x) for x in itertools.combinations([
                'CP_ratio', 'EP_ratio', 'MV_log', 'SP_ratio', 'BM_ratio', 
                'EP_surprise', 'SP_surprise', 
                'EP_surprise_std', 'EP_surprise_mean_std', 'First_Ana_Pos'], 2)]
        }



MODELS = [
        #LogisticRegressionCV(solver = "lbfgs", max_iter = 1000, 
        #                     cv = 5, Cs = 100, n_jobs = -1),
        RandomForestRegressor(n_estimators = 2500, max_depth = 15, 
                              random_state = 0, n_jobs = -1)
        ]


ml_jobs = [{
       'model': m,
       'features': BASE_FEATURES,
        'train_target': t,
        'return_target': t,
       }  for t in TARGETS for m in MODELS]



MODEL_JOBS = ml_jobs

MODEL_JOBS = [{**j, **GENERAL_MODEL_CONFIG} for j in MODEL_JOBS]


def generate_model_jobs():
    return MODEL_JOBS

def generate_backtest_jobs():
    return []


