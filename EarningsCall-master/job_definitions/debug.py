#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 08:13:22 2019

@author: mschnaubelt
"""

import pandas as pd
from backtest.strategy import call_strategy

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from util.validator import PhysicalTimeForwardValidation

from models.earnings_surprise import EarningsSurpriseModel


HOLDING_PERIOD = 5



GENERAL_MODEL_CONFIG = {
        'train_subset': 'SP1500',
        'train_target': 'abnormal_%dd_drift' % HOLDING_PERIOD,
        'return_target': 'abnormal_%dd_drift' % HOLDING_PERIOD,
        'top_flop_cutoff': 0.1,
        'validator': PhysicalTimeForwardValidation('2013-01-01', pd.Timedelta(12, 'M'), 
                                                   1500, 'final_datetime'),
        'rolling_window_size': 1500,
        }


BASE_FEATURES = ['earnings_surprise', 'earnings_surprise',
                 'earnings_surprise_std', 'earnings_surprise_count', 'earnings_ratio', 
                 'log_length', 'nr_analysts', 'pays_dividend', 
                 'general_PositivityLM', 'general_NegativityLM', 
                 'qanda_PositivityLM', 'qanda_NegativityLM'
                 ]

#BASE_FEATURES += ['%d_pos' % i for i in range(30)]
#BASE_FEATURES += ['%d_neg' % i for i in range(30)]

#BASE_FEATURES += ['q_%d_pos' % i for i in range(30)]
#BASE_FEATURES += ['q_%d_neg' % i for i in range(30)]

#BASE_FEATURES += ['5_pos','5_neg','9_pos','9_neg','10_pos','10_neg','12_pos','12_neg','15_pos','15_neg','19_pos','19_neg','25_pos','25_neg','28_pos','28_neg','29_pos','29_neg','q_6_pos','q_6_neg','q_7_pos','q_7_neg','q_14_pos','q_14_neg','q_20_pos','q_20_neg','q_24_pos','q_24_neg']


rf_job = {
       'model': RandomForestClassifier(n_estimators = 1000, max_depth = 10, min_samples_split = 200,
                                       class_weight = "balanced_subsample", 
                                       random_state = 0, n_jobs = -1),
       'features': BASE_FEATURES,
       'calculate_permutation_feature_importances': False
       }



MODEL_JOBS = [rf_job]

MODEL_JOBS = [{**j, **GENERAL_MODEL_CONFIG} for j in MODEL_JOBS]


steps = [('scaler', StandardScaler()), ('ANN', MLPClassifier(
        hidden_layer_sizes = (50, 25, 10), activation = 'tanh',
        max_iter= 5000))]


{
       'train_subset': 'SP1500',
       'model': RandomForestClassifier(n_estimators = 1000, max_depth = 6, min_samples_leaf = 100,
                                       class_weight = "balanced_subsample", 
                                       random_state = 0, n_jobs = -1),
       #'model': LogisticRegression(solver="lbfgs", max_iter= 5000),
       #'model': Pipeline(steps),
       #'model': EarningsSurpriseModel('earnings_surprise'),
       'train_target': 'abnormal_%dd_drift' % HOLDING_PERIOD,
       'return_target': 'abnormal_%dd_drift' % HOLDING_PERIOD,
       'features': ['earnings_surprise', 'earnings_surprise', 
                    'log_length', 
                    'nr_analysts', #'nr_executives',
                    'pays_dividend', 
                    #'same_day_call_count_ge_100', 
                    'hour_of_day_half', 
                    'general_PositivityLM', 'general_NegativityLM', 
                    'qanda_PositivityLM', 'qanda_NegativityLM',
                    #'general_RatioUncertaintyLM', 
                    #'qanda_RatioUncertaintyLM',
                    #'whole_general_neg', 'whole_general_pos', 'whole_general_unc',
                    #'abnormal_call_return'
                    ],
        'top_flop_cutoff': 0.1,
        'validator': PhysicalTimeForwardValidation('2013-01-01', pd.Timedelta(12, 'M'), 
                                                   1500, 'final_datetime'),
        'rolling_window_size': 1500,
        'calculate_permutation_feature_importances': False
       }


#model = lightgbm.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, 
#                                learning_rate=0.001, n_estimators=1000, subsample_for_bin=200000, 
#                                objective=None, class_weight='balanced', min_split_gain=0.0, 
#                                min_child_weight=0.001, min_child_samples=100, subsample=1.0, 
#                                subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, 
#                                reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True, 
#                                importance_type='split')



GENERAL_BACKTEST_CONFIG = {
            'commission': 0.001,
            'market_commission': 0.0001,
            #'backtest_subset': 'SP500TR',
            'start': pd.Timestamp('2013-01-01', tz = 'America/New_York'),
            'end': pd.Timestamp('2018-12-31', tz = 'America/New_York')
        }


BACKTEST_JOBS = [
        {
        'name': 'per-call-%dd-0.1-LS' % HOLDING_PERIOD,
        'strategy': call_strategy,
        'strategy_args': {
                'min_rank': 0.1,
                'holding_days': HOLDING_PERIOD,
                'allocation': 'per_call',
                'per_call_fraction': 0.02,
                'long_short': 'long_short'
                }
        },
        {
        'name': 'per-day-%dd-0.1-LS' % HOLDING_PERIOD,
        'strategy': call_strategy,
        'strategy_args': {
                'min_rank': 0.1,
                'holding_days': HOLDING_PERIOD,
                'allocation': 'per_day',
                'long_short': 'long_short'
                }
        }
        ]

BACKTEST_JOBS = [{**j, **GENERAL_BACKTEST_CONFIG} for j in BACKTEST_JOBS]



def generate_model_jobs():
    return MODEL_JOBS

def generate_backtest_jobs():
    return BACKTEST_JOBS


