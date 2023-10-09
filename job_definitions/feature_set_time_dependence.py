#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 08:13:22 2019

@author: mschnaubelt
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from util.validator import PhysicalTimeForwardValidation

from config import RATIO_FEATURES, FORECAST_ERROR_FEATURES, DISPERSION_FEATURES, POL_FEATURES



TARGETS = [
           'ff-dec_abnormal_-5d_drift',
           'ff-dec_abnormal_-1d_drift',
           
           'ff-dec_abnormal_1d_drift',
           'ff-dec_abnormal_2d_drift',
           'ff-dec_abnormal_3d_drift',
           'ff-dec_abnormal_4d_drift',
           'ff-dec_abnormal_5d_drift',
           'ff-dec_abnormal_10d_drift',
           'ff-dec_abnormal_15d_drift',
           'ff-dec_abnormal_20d_drift',
           'ff-dec_abnormal_30d_drift',
           'ff-dec_abnormal_40d_drift',
           'ff-dec_abnormal_50d_drift',
           'ff-dec_abnormal_60d_drift',
           
           'ff-dec_abnormal_9d_drift',
           'ff-dec_abnormal_25d_drift',
           'ff-dec_abnormal_35d_drift',
           'ff-dec_abnormal_45d_drift',
           'ff-dec_abnormal_55d_drift',
           'ff-dec_abnormal_-4d_drift',
           'ff-dec_abnormal_-3d_drift',
           'ff-dec_abnormal_-2d_drift'
          ]


GENERAL_MODEL_CONFIG = {
        'train_subset': 'SP1500',
        'top_flop_cutoff': 0.1,
        'validator': PhysicalTimeForwardValidation('2013-01-01', pd.Timedelta(12, 'M'), 
                                                   1500, 'final_datetime'),
        'rolling_window_size': 1500,
        'model': RandomForestRegressor(n_estimators = 5000, max_depth = 20, 
                                        random_state = 0, n_jobs = -1),
        'calculate_permutation_feature_importances': False,
        'calculate_partial_dependence': False
        }



FEATURE_SETS = [
        RATIO_FEATURES, 
        FORECAST_ERROR_FEATURES, 
        DISPERSION_FEATURES, 
        POL_FEATURES,
        RATIO_FEATURES + FORECAST_ERROR_FEATURES + DISPERSION_FEATURES + POL_FEATURES
                ]


MODEL_JOBS = [{
        'features': f, 
        'train_target': t, 
        'return_target': t,
            } for t in TARGETS for f in FEATURE_SETS]


MODEL_JOBS = [{**j, **GENERAL_MODEL_CONFIG} for j in MODEL_JOBS]



def generate_model_jobs():
    return MODEL_JOBS

def generate_backtest_jobs():
    return []


