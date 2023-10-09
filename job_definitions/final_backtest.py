#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 08:13:22 2019

@author: mschnaubelt
"""

import itertools
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer

from backtest.strategy import call_strategy, buy_and_hold_strategy
from util.validator import PhysicalTimeForwardValidation
from models.earnings_surprise import EarningsSurpriseRegressor

from config import RATIO_FEATURES, FORECAST_ERROR_FEATURES, DISPERSION_FEATURES, POL_FEATURES


HOLDING_PERIOD = 5
MIN_RANK = 0.1

TARGETS = ['ff-dec_abnormal_%dd_drift' % HOLDING_PERIOD]



GENERAL_MODEL_CONFIG = {
        'train_subset': 'SP1500',
        'top_flop_cutoff': MIN_RANK,
        'validator': PhysicalTimeForwardValidation('2013-01-01', pd.Timedelta(12, 'M'), 
                                                   1500, 'final_datetime'),
        'rolling_window_size': 1500,
        'calculate_permutation_feature_importances': False,
        'calculate_partial_dependence': False
        }


ADD_FEATURES = list(itertools.product([POL_FEATURES],
                                      [DISPERSION_FEATURES],
                                      [FORECAST_ERROR_FEATURES], 
                                      [RATIO_FEATURES]
                                      ))

ADD_FEATURES = list(filter(lambda x: len(x)>0, 
                           map(lambda l: l[0] + l[1] + l[2] + l[3], ADD_FEATURES)))



MODELS = [
        EarningsSurpriseRegressor('EP_surprise'),
        LinearRegression(n_jobs = -1),
        Pipeline([('scaler', KBinsDiscretizer(n_bins = 10)), 
                  ('LR', LinearRegression(n_jobs = -1))]),
        RandomForestRegressor(n_estimators = 5000, max_depth = 18, 
                              random_state = 0, n_jobs = -1)
        ]


MODEL_JOBS = [{
       'model': m,
       'features': ef,
        'train_target': t,
        'return_target': t,
       }  for t in TARGETS for m in MODELS for ef in ADD_FEATURES]


MODEL_JOBS = [{**j, **GENERAL_MODEL_CONFIG} for j in MODEL_JOBS]



BACKTEST_JOBS = [
        {
            'name': 'buy-and-hold',
            'strategy': buy_and_hold_strategy,
            'strategy_args': {
                    },
            'default_commission': 0.001,
            'market_segment_commission': {'SP500TR': 0.001, 'SP400TR': 0.0015, 'SP600TR': 0.002},
            'market_commission': 0.0001,
            'start': pd.Timestamp('2013-01-01', tz = 'America/New_York'),
            'end': pd.Timestamp('2019-03-31', tz = 'America/New_York')
        },
#        {
#        'name': 'per-call-%dd-0.1-LS' % HOLDING_PERIOD,
#        'strategy': call_strategy,
#        'strategy_args': {
#                'min_rank': MIN_RANK,
#                'holding_days': HOLDING_PERIOD,
#                'allocation': 'per_call',
#                'per_call_fraction': 0.02,
#                'long_short': 'long_short'
#                }
#        },
#        {
#        'name': 'per-call-%dd-0.1-L' % HOLDING_PERIOD,
#        'strategy': call_strategy,
#        'strategy_args': {
#                'min_rank': MIN_RANK,
#                'holding_days': HOLDING_PERIOD,
#                'allocation': 'per_call',
#                'per_call_fraction': 0.02,
#                'long_short': 'long'
#                }
#        },
        {
            'name': 'per-day-%dd-0.1-LS' % HOLDING_PERIOD,
            'strategy': call_strategy,
            'strategy_args': {
                    'min_rank': MIN_RANK,
                    'holding_days': HOLDING_PERIOD,
                    'allocation': 'per_day',
                    'long_short': 'long_short'
                    },
            'default_commission': 0.001,
            'market_segment_commission': {'SP500TR': 0.001, 'SP400TR': 0.0015, 'SP600TR': 0.002},
            'market_commission': 0.0001,
            'start': pd.Timestamp('2013-01-01', tz = 'America/New_York'),
            'end': pd.Timestamp('2019-03-31', tz = 'America/New_York')
        },
        {
            'name': 'per-day-%dd-0.1-LS-noTAC' % HOLDING_PERIOD,
            'strategy': call_strategy,
            'strategy_args': {
                    'min_rank': MIN_RANK,
                    'holding_days': HOLDING_PERIOD,
                    'allocation': 'per_day',
                    'long_short': 'long_short'
                    },
            'default_commission': 0.0,
            'market_segment_commission': {'SP500TR': 0.0, 'SP400TR': 0.0, 'SP600TR': 0.0},
            'market_commission': 0.0,
            'start': pd.Timestamp('2013-01-01', tz = 'America/New_York'),
            'end': pd.Timestamp('2019-03-31', tz = 'America/New_York')
        },
#        {
#        'name': 'per-day-%dd-0.1-L' % HOLDING_PERIOD,
#        'strategy': call_strategy,
#        'strategy_args': {
#                'min_rank': MIN_RANK,
#                'holding_days': HOLDING_PERIOD,
#                'allocation': 'per_day',
#                'long_short': 'long'
#                }
#        }
        ]



def generate_model_jobs():
    return MODEL_JOBS

def generate_backtest_jobs():
    return BACKTEST_JOBS


