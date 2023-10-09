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


GENERAL_MODEL_CONFIG = {
        'train_subset': 'SP1500',
        'top_flop_cutoff': 0.1,
        'features': BASE_FEATURES,
        'validator': PhysicalTimeForwardValidation('2013-01-01', pd.Timedelta(12, 'M'), 
                                                   1500, 'final_datetime'),
        'calculate_permutation_feature_importances': False,
        'calculate_partial_dependence': False
        }


PARAM_CONFIGS = [
        {
                'model': RandomForestRegressor(n_estimators = 5000, max_depth = 20, 
                                               random_state = 0, n_jobs = -1),
                'rolling_window_size': 1500
        },
        
        {
                'model': RandomForestRegressor(n_estimators = 5000, max_depth = 30, 
                                               random_state = 0, n_jobs = -1),
                'rolling_window_size': 1500
        },
        {
                'model': RandomForestRegressor(n_estimators = 5000, max_depth = 10, 
                                               random_state = 0, n_jobs = -1),
                'rolling_window_size': 1500
        },
        
        {
                'model': RandomForestRegressor(n_estimators = 10000, max_depth = 20, 
                                               random_state = 0, n_jobs = -1),
                'rolling_window_size': 1500
        },
        {
                'model': RandomForestRegressor(n_estimators = 2500, max_depth = 20, 
                                               random_state = 0, n_jobs = -1),
                'rolling_window_size': 1500
        },
        
        {
                'model': RandomForestRegressor(n_estimators = 5000, max_depth = 20, 
                                               random_state = 0, n_jobs = -1),
                'rolling_window_size': 500
        },
        {
                'model': RandomForestRegressor(n_estimators = 5000, max_depth = 20, 
                                               random_state = 0, n_jobs = -1),
                'rolling_window_size': 1000
        }
        ]



MODEL_JOBS = [{**c, **GENERAL_MODEL_CONFIG, 'train_target': t, 'return_target': t} 
                for t in TARGETS for c in PARAM_CONFIGS]



def generate_model_jobs():
    return MODEL_JOBS

def generate_backtest_jobs():
    return []


