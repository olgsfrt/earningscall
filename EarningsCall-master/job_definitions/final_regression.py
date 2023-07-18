#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 08:13:22 2019

@author: mschnaubelt
"""

import itertools
import pandas as pd
import numpy as np
from backtest.strategy import call_strategy, buy_and_hold_strategy

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, KBinsDiscretizer, FunctionTransformer

from models.earnings_surprise import EarningsSurpriseRegressor
from util.validator import PhysicalTimeForwardValidation

from config import RATIO_FEATURES, TRAILING_RATIO_FEATURES, FORECAST_ERROR_FEATURES, DISPERSION_FEATURES, POL_FEATURES, BASE_FEATURES


HOLDING_PERIOD = 5
#TARGETS = ['ff5_abnormal_5d_drift', 'ff5_abnormal_20d_drift', 'ff5_abnormal_60d_drift']
TARGETS = ['ff-dec_abnormal_-5d_drift', 'ff-dec_abnormal_5d_drift', 'ff-dec_abnormal_20d_drift', 'ff-dec_abnormal_60d_drift']
#TARGETS = ['abnormal_5d_drift', 'abnormal_20d_drift', 'abnormal_60d_drift']
MIN_RANK = 0.1



GENERAL_MODEL_CONFIG = {
        'train_subset': 'SP1500',
        'top_flop_cutoff': MIN_RANK,
        'validator': PhysicalTimeForwardValidation('2013-01-01', pd.Timedelta(12, 'M'), 
                                                   1500, 'final_datetime'),
        'rolling_window_size': 1500,
        'calculate_permutation_feature_importances': False,
        'calculate_partial_dependence': False
        }


ADD_FEATURES = list(itertools.product([[], POL_FEATURES], #[f + ':roll-z' for f in POL_FEATURES]
                                      [[], DISPERSION_FEATURES],
                                      [[], FORECAST_ERROR_FEATURES], 
                                      [[], RATIO_FEATURES]#, TRAILING_RATIO_FEATURES]
                                      ))

ADD_FEATURES = list(filter(lambda x: len(x)>0, 
                           map(lambda l: l[0] + l[1] + l[2] + l[3], ADD_FEATURES)))

ADD_FEATURES = [
        RATIO_FEATURES + FORECAST_ERROR_FEATURES + DISPERSION_FEATURES + POL_FEATURES,
        RATIO_FEATURES, FORECAST_ERROR_FEATURES, #DISPERSION_FEATURES, POL_FEATURES,
        #RATIO_FEATURES + FORECAST_ERROR_FEATURES
        ]


es_jobs = [{
       'model': EarningsSurpriseRegressor('EP_surprise'),
       'features': ['EP_surprise'],
        'train_target': t,
        'return_target': t,
       } for t in TARGETS]

rs_jobs = [{
       'model': EarningsSurpriseRegressor('EP_ratio'),
       'features': ['EP_ratio'],
        'train_target': t,
        'return_target': t,
       } for t in TARGETS]


def add_quadratic(X):
    return np.concatenate((X, X**2), axis = 1)

def add_quadratic_tanh(X):
    return np.concatenate((X, X**2, np.tanh(X)), axis = 1)

MODELS = [
        LinearRegression(n_jobs = -1),
        LassoCV(max_iter = 2000, random_state = 0, cv = 5, n_alphas = 100, n_jobs = -1),
        #ElasticNetCV(max_iter = 2000, random_state = 0,
        #             l1_ratio = [.1, .5, .7, .9, .95, .99, 1],
        #             cv = 5, n_alphas = 20, n_jobs = -1),
        Pipeline([('scaler', KBinsDiscretizer(n_bins = 5)), 
                  ('LR', LinearRegression(n_jobs = -1))]),
        Pipeline([('scaler', KBinsDiscretizer(n_bins = 10)), 
                  ('LR', LinearRegression(n_jobs = -1))]),
        Pipeline([('scaler', KBinsDiscretizer(n_bins = 10)), 
                  ('LAS', LassoCV(max_iter = 2000, random_state = 0, cv = 5, n_alphas = 100, n_jobs = -1))]),
        Pipeline([('scaler', KBinsDiscretizer(n_bins = 20)), 
                  ('LR', LinearRegression(n_jobs = -1))]),
        Pipeline([('scaler', QuantileTransformer()), 
                  ('LR', LinearRegression(n_jobs = -1))]),
        Pipeline([('scaler', QuantileTransformer()), 
                  ('LAS', LassoCV(max_iter = 2000, random_state = 0, cv = 5, n_alphas = 100, n_jobs = -1))]),
        Pipeline([('scaler', FunctionTransformer(add_quadratic, validate=True)), 
                  ('LR', LinearRegression(n_jobs = -1))]),
        Pipeline([('scaler', FunctionTransformer(add_quadratic_tanh, validate=True)), 
                  ('LR', LinearRegression(n_jobs = -1))]),
        RandomForestRegressor(n_estimators = 5000, max_depth = 20, 
                              random_state = 0, n_jobs = -1)
        ]


ml_jobs = [{
       'model': m,
       'features': ef,
        'train_target': t,
        'return_target': t,
       }  for t in TARGETS for m in MODELS for ef in ADD_FEATURES]



MODEL_JOBS = es_jobs + ml_jobs #+ rs_jobs

MODEL_JOBS = [{**j, **GENERAL_MODEL_CONFIG} for j in MODEL_JOBS]


GENERAL_BACKTEST_CONFIG = {
            'default_commission': 0.001,
            'market_segment_commission': {'SP500TR': 0.001, 'SP400TR': 0.0015, 'SP600TR': 0.002},
            'market_commission': 0.0001,
            'start': pd.Timestamp('2013-01-01', tz = 'America/New_York'),
            'end': pd.Timestamp('2018-12-31', tz = 'America/New_York')
        }


BACKTEST_JOBS = [
        {
        'name': 'per-day-%dd-0.1-LS' % HOLDING_PERIOD,
        'strategy': call_strategy,
        'strategy_args': {
                'min_rank': MIN_RANK,
                'holding_days': HOLDING_PERIOD,
                'allocation': 'per_day',
                'long_short': 'long_short'
                }
        },
        {
        'name': 'per-day-%dd-0.1-L' % HOLDING_PERIOD,
        'strategy': call_strategy,
        'strategy_args': {
                'min_rank': MIN_RANK,
                'holding_days': HOLDING_PERIOD,
                'allocation': 'per_day',
                'long_short': 'long'
                }
        }
        ]

BACKTEST_JOBS = [{**j, **GENERAL_BACKTEST_CONFIG} for j in BACKTEST_JOBS]



def generate_model_jobs():
    return MODEL_JOBS

def generate_backtest_jobs():
    return BACKTEST_JOBS


