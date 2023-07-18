#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 08:13:22 2019

@author: mschnaubelt
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from util.validator import PhysicalTimeForwardValidation
from config import BASE_FEATURES


TARGETS = ['ff-dec_abnormal_5d_drift', 'ff-dec_abnormal_60d_drift']
MIN_RANK = 0.1



GENERAL_MODEL_CONFIG = {
        'train_subset': 'SP1500',
        'top_flop_cutoff': MIN_RANK,
        'features': BASE_FEATURES,
        'validator': PhysicalTimeForwardValidation('2013-01-01', pd.Timedelta(12, 'M'), 
                                                   1500, 'final_datetime'),
        'rolling_window_size': 1500,
        'calculate_permutation_feature_importances': False,
        'calculate_partial_dependence': False
        }


MODELS = [
        RandomForestRegressor(n_estimators = 5000, max_depth = 20, 
                              random_state = i, n_jobs = -1) for i in range(10)
        ]


MODEL_JOBS = [{
       'model': m,
       'train_target': t,
       'return_target': t,
       }  for t in TARGETS for m in MODELS]


MODEL_JOBS = [{**j, **GENERAL_MODEL_CONFIG} for j in MODEL_JOBS]



def generate_model_jobs():
    return MODEL_JOBS

def generate_backtest_jobs():
    return []


