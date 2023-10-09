#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 08:13:22 2019

@author: mschnaubelt
"""

import os

os.chdir('c:\\Users\\aq75iwit\\Anaconda3\\envs\\earnings_call_7\\EarningsCall')

os.getcwd()

#'c:\\Users\\aq75iwit\\Anaconda3\\envs\\earnings_call_7\\EarningsCall'

import itertools
import pandas as pd
#from backtest.strategy import call_strategy, buy_and_hold_strategy

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from util.validator import PhysicalTimeForwardValidation

from models.earnings_surprise import EarningsSurpriseModel

from config import RATIO_FEATURES, TRAILING_RATIO_FEATURES, FORECAST_ERROR_FEATURES, DISPERSION_FEATURES, POL_FEATURES


HOLDING_PERIOD = 5
#TARGETS = ['ff5_abnormal_5d_drift', 'ff5_abnormal_20d_drift', 'ff5_abnormal_60d_drift']
TARGETS = ['ff5_abnormal_20d_drift']
#TARGETS = ['ff-dec_abnormal_-5d_drift', 'ff-dec_abnormal_5d_drift', 'ff-dec_abnormal_20d_drift', 'ff-dec_abnormal_60d_drift']
#TARGETS = ['abnormal_5d_drift', 'abnormal_20d_drift', 'abnormal_60d_drift']
MIN_RANK = 0.1


#GENERAL_MODEL_CONFIG = {
#    'train_subset': 'SP1500',
#    'top_flop_cutoff': MIN_RANK,
#    'validator': PhysicalTimeForwardValidation('2013-01-01', pd.Timedelta(days=365), 1500, 'final_datetime'),
#    'rolling_window_size': 1500,
#    'calculate_permutation_feature_importances': False,
#    'calculate_partial_dependence': False
#}

GENERAL_MODEL_CONFIG = {
    'train_subset': 'SP500TR',
    'top_flop_cutoff': MIN_RANK,
    'validator': PhysicalTimeForwardValidation('2007-11-01', pd.Timedelta(days=180), 100, 'final_datetime'),
    'rolling_window_size': 100,
    'calculate_permutation_feature_importances': True,
    'calculate_partial_dependence': False
}





#ANA_FEATURES = ['First_Ana_Pos','First_Ana_Neg','Last_Ana_Pos','Last_Ana_Neg']
#EA_FEATURES = ['LM_Positivity_Analysts','LM_Negativity_Analysts','LM_Positivity_Executives','LM_Negativity_Executives']
#
#ALL_TS = ['assets_LM_Positivity', 'assets_LM_Negativity', 'liabilities_LM_Positivity', 'liabilities_LM_Negativity', 'equity_LM_Positivity', 'equity_LM_Negativity', 'revenues_LM_Positivity', 'revenues_LM_Negativity', 'expenses_LM_Positivity', 'expenses_LM_Negativity', 'loss_LM_Positivity', 'loss_LM_Negativity', 'earnings_LM_Positivity', 'earnings_LM_Negativity', 'mergers_LM_Positivity', 'mergers_LM_Negativity', 'strategy_LM_Positivity', 'strategy_LM_Negativity', 'innovation_LM_Positivity', 'innovation_LM_Negativity', 'f_markets_LM_Positivity', 'f_markets_LM_Negativity', 'distribution_LM_Positivity', 'distribution_LM_Negativity', 'outlook_LM_Positivity', 'outlook_LM_Negativity', 'environment_LM_Positivity', 'environment_LM_Negativity', 'liquidity_LM_Positivity', 'liquidity_LM_Negativity', 'movements_LM_Positivity', 'movements_LM_Negativity']
#OLEG_TS = ['assets_LM_Positivity', 'assets_LM_Negativity', 'revenues_LM_Positivity', 'revenues_LM_Negativity', 'expenses_LM_Positivity', 'expenses_LM_Negativity', 'earnings_LM_Positivity', 'earnings_LM_Negativity', 'innovation_LM_Positivity', 'innovation_LM_Negativity', 'outlook_LM_Positivity', 'outlook_LM_Negativity', 'environment_LM_Positivity', 'environment_LM_Negativity', 'movements_LM_Positivity', 'movements_LM_Negativity']
#MINI_TS = ['outlook_q_s_LM_Positivity', 'outlook_q_s_LM_Negativity', 'distribution_q_s_LM_Positivity', 'distribution_q_s_LM_Negativity', 'movements_q_s_LM_Positivity', 'movements_q_s_LM_Negativity']
#
#
#LDA_INTRO = ['intro_topic_0_LM_Sentiment', 'intro_topic_0_LM_Positivity', 'intro_topic_0_LM_Negativity', 'intro_topic_0_LM_Uncertainty', 'intro_topic_1_LM_Sentiment', 'intro_topic_1_LM_Positivity', 'intro_topic_1_LM_Negativity', 'intro_topic_1_LM_Uncertainty', 'intro_topic_2_LM_Sentiment', 'intro_topic_2_LM_Positivity', 'intro_topic_2_LM_Negativity', 'intro_topic_2_LM_Uncertainty', 'intro_topic_3_LM_Sentiment', 'intro_topic_3_LM_Positivity', 'intro_topic_3_LM_Negativity', 'intro_topic_3_LM_Uncertainty', 'intro_topic_4_LM_Sentiment', 'intro_topic_4_LM_Positivity', 'intro_topic_4_LM_Negativity', 'intro_topic_4_LM_Uncertainty', 'intro_topic_5_LM_Sentiment', 'intro_topic_5_LM_Positivity', 'intro_topic_5_LM_Negativity', 'intro_topic_5_LM_Uncertainty', 'intro_topic_6_LM_Sentiment', 'intro_topic_6_LM_Positivity', 'intro_topic_6_LM_Negativity', 'intro_topic_6_LM_Uncertainty', 'intro_topic_7_LM_Sentiment', 'intro_topic_7_LM_Positivity', 'intro_topic_7_LM_Negativity', 'intro_topic_7_LM_Uncertainty', 'intro_topic_8_LM_Sentiment', 'intro_topic_8_LM_Positivity', 'intro_topic_8_LM_Negativity', 'intro_topic_8_LM_Uncertainty', 'intro_topic_9_LM_Sentiment', 'intro_topic_9_LM_Positivity', 'intro_topic_9_LM_Negativity', 'intro_topic_9_LM_Uncertainty', 'intro_topic_10_LM_Sentiment', 'intro_topic_10_LM_Positivity', 'intro_topic_10_LM_Negativity', 'intro_topic_10_LM_Uncertainty', 'intro_topic_11_LM_Sentiment', 'intro_topic_11_LM_Positivity', 'intro_topic_11_LM_Negativity', 'intro_topic_11_LM_Uncertainty', 'intro_topic_12_LM_Sentiment', 'intro_topic_12_LM_Positivity', 'intro_topic_12_LM_Negativity', 'intro_topic_12_LM_Uncertainty', 'intro_topic_13_LM_Sentiment', 'intro_topic_13_LM_Positivity', 'intro_topic_13_LM_Negativity', 'intro_topic_13_LM_Uncertainty', 'intro_topic_14_LM_Sentiment', 'intro_topic_14_LM_Positivity', 'intro_topic_14_LM_Negativity', 'intro_topic_14_LM_Uncertainty', 'intro_topic_15_LM_Sentiment', 'intro_topic_15_LM_Positivity', 'intro_topic_15_LM_Negativity', 'intro_topic_15_LM_Uncertainty', 'intro_topic_16_LM_Sentiment', 'intro_topic_16_LM_Positivity', 'intro_topic_16_LM_Negativity', 'intro_topic_16_LM_Uncertainty', 'intro_topic_17_LM_Sentiment', 'intro_topic_17_LM_Positivity', 'intro_topic_17_LM_Negativity', 'intro_topic_17_LM_Uncertainty', 'intro_topic_18_LM_Sentiment', 'intro_topic_18_LM_Positivity', 'intro_topic_18_LM_Negativity', 'intro_topic_18_LM_Uncertainty', 'intro_topic_19_LM_Sentiment', 'intro_topic_19_LM_Positivity', 'intro_topic_19_LM_Negativity', 'intro_topic_19_LM_Uncertainty', 'intro_topic_20_LM_Sentiment', 'intro_topic_20_LM_Positivity', 'intro_topic_20_LM_Negativity', 'intro_topic_20_LM_Uncertainty', 'intro_topic_21_LM_Sentiment', 'intro_topic_21_LM_Positivity', 'intro_topic_21_LM_Negativity', 'intro_topic_21_LM_Uncertainty', 'intro_topic_22_LM_Sentiment', 'intro_topic_22_LM_Positivity', 'intro_topic_22_LM_Negativity', 'intro_topic_22_LM_Uncertainty', 'intro_topic_23_LM_Sentiment', 'intro_topic_23_LM_Positivity', 'intro_topic_23_LM_Negativity', 'intro_topic_23_LM_Uncertainty', 'intro_topic_24_LM_Sentiment', 'intro_topic_24_LM_Positivity', 'intro_topic_24_LM_Negativity', 'intro_topic_24_LM_Uncertainty', 'intro_topic_25_LM_Sentiment', 'intro_topic_25_LM_Positivity', 'intro_topic_25_LM_Negativity', 'intro_topic_25_LM_Uncertainty', 'intro_topic_26_LM_Sentiment', 'intro_topic_26_LM_Positivity', 'intro_topic_26_LM_Negativity', 'intro_topic_26_LM_Uncertainty', 'intro_topic_27_LM_Sentiment', 'intro_topic_27_LM_Positivity', 'intro_topic_27_LM_Negativity', 'intro_topic_27_LM_Uncertainty', 'intro_topic_28_LM_Sentiment', 'intro_topic_28_LM_Positivity', 'intro_topic_28_LM_Negativity', 'intro_topic_28_LM_Uncertainty', 'intro_topic_29_LM_Sentiment', 'intro_topic_29_LM_Positivity', 'intro_topic_29_LM_Negativity', 'intro_topic_29_LM_Uncertainty']
#LDA_QANDA = ['qanda_topic_0_LM_Sentiment', 'qanda_topic_0_LM_Positivity', 'qanda_topic_0_LM_Negativity', 'qanda_topic_0_LM_Uncertainty', 'qanda_topic_1_LM_Sentiment', 'qanda_topic_1_LM_Positivity', 'qanda_topic_1_LM_Negativity', 'qanda_topic_1_LM_Uncertainty', 'qanda_topic_2_LM_Sentiment', 'qanda_topic_2_LM_Positivity', 'qanda_topic_2_LM_Negativity', 'qanda_topic_2_LM_Uncertainty', 'qanda_topic_3_LM_Sentiment', 'qanda_topic_3_LM_Positivity', 'qanda_topic_3_LM_Negativity', 'qanda_topic_3_LM_Uncertainty', 'qanda_topic_4_LM_Sentiment', 'qanda_topic_4_LM_Positivity', 'qanda_topic_4_LM_Negativity', 'qanda_topic_4_LM_Uncertainty', 'qanda_topic_5_LM_Sentiment', 'qanda_topic_5_LM_Positivity', 'qanda_topic_5_LM_Negativity', 'qanda_topic_5_LM_Uncertainty', 'qanda_topic_6_LM_Sentiment', 'qanda_topic_6_LM_Positivity', 'qanda_topic_6_LM_Negativity', 'qanda_topic_6_LM_Uncertainty', 'qanda_topic_7_LM_Sentiment', 'qanda_topic_7_LM_Positivity', 'qanda_topic_7_LM_Negativity', 'qanda_topic_7_LM_Uncertainty', 'qanda_topic_8_LM_Sentiment', 'qanda_topic_8_LM_Positivity', 'qanda_topic_8_LM_Negativity', 'qanda_topic_8_LM_Uncertainty', 'qanda_topic_9_LM_Sentiment', 'qanda_topic_9_LM_Positivity', 'qanda_topic_9_LM_Negativity', 'qanda_topic_9_LM_Uncertainty', 'qanda_topic_10_LM_Sentiment', 'qanda_topic_10_LM_Positivity', 'qanda_topic_10_LM_Negativity', 'qanda_topic_10_LM_Uncertainty', 'qanda_topic_11_LM_Sentiment', 'qanda_topic_11_LM_Positivity', 'qanda_topic_11_LM_Negativity', 'qanda_topic_11_LM_Uncertainty', 'qanda_topic_12_LM_Sentiment', 'qanda_topic_12_LM_Positivity', 'qanda_topic_12_LM_Negativity', 'qanda_topic_12_LM_Uncertainty', 'qanda_topic_13_LM_Sentiment', 'qanda_topic_13_LM_Positivity', 'qanda_topic_13_LM_Negativity', 'qanda_topic_13_LM_Uncertainty', 'qanda_topic_14_LM_Sentiment', 'qanda_topic_14_LM_Positivity', 'qanda_topic_14_LM_Negativity', 'qanda_topic_14_LM_Uncertainty']
#
#LDA_INTRO_PN = ['intro_topic_0_LM_Positivity', 'intro_topic_0_LM_Negativity', 'intro_topic_1_LM_Positivity', 'intro_topic_1_LM_Negativity', 'intro_topic_2_LM_Positivity', 'intro_topic_2_LM_Negativity', 'intro_topic_3_LM_Positivity', 'intro_topic_3_LM_Negativity', 'intro_topic_4_LM_Positivity', 'intro_topic_4_LM_Negativity', 'intro_topic_5_LM_Positivity', 'intro_topic_5_LM_Negativity', 'intro_topic_6_LM_Positivity', 'intro_topic_6_LM_Negativity', 'intro_topic_7_LM_Positivity', 'intro_topic_7_LM_Negativity', 'intro_topic_8_LM_Positivity', 'intro_topic_8_LM_Negativity', 'intro_topic_9_LM_Positivity', 'intro_topic_9_LM_Negativity', 'intro_topic_10_LM_Positivity', 'intro_topic_10_LM_Negativity', 'intro_topic_11_LM_Positivity', 'intro_topic_11_LM_Negativity', 'intro_topic_12_LM_Positivity', 'intro_topic_12_LM_Negativity', 'intro_topic_13_LM_Positivity', 'intro_topic_13_LM_Negativity', 'intro_topic_14_LM_Positivity', 'intro_topic_14_LM_Negativity', 'intro_topic_15_LM_Positivity', 'intro_topic_15_LM_Negativity', 'intro_topic_16_LM_Positivity', 'intro_topic_16_LM_Negativity', 'intro_topic_17_LM_Positivity', 'intro_topic_17_LM_Negativity', 'intro_topic_18_LM_Positivity', 'intro_topic_18_LM_Negativity', 'intro_topic_19_LM_Positivity', 'intro_topic_19_LM_Negativity', 'intro_topic_20_LM_Positivity', 'intro_topic_20_LM_Negativity', 'intro_topic_21_LM_Positivity', 'intro_topic_21_LM_Negativity', 'intro_topic_22_LM_Positivity', 'intro_topic_22_LM_Negativity', 'intro_topic_23_LM_Positivity', 'intro_topic_23_LM_Negativity', 'intro_topic_24_LM_Positivity', 'intro_topic_24_LM_Negativity', 'intro_topic_25_LM_Positivity', 'intro_topic_25_LM_Negativity', 'intro_topic_26_LM_Positivity', 'intro_topic_26_LM_Negativity', 'intro_topic_27_LM_Positivity', 'intro_topic_27_LM_Negativity', 'intro_topic_28_LM_Positivity', 'intro_topic_28_LM_Negativity', 'intro_topic_29_LM_Positivity', 'intro_topic_29_LM_Negativity', 'intro_topic_29_LM_Uncertainty']
#LDA_QANDA_PN = ['qanda_topic_0_LM_Positivity', 'qanda_topic_0_LM_Negativity', 'qanda_topic_1_LM_Positivity', 'qanda_topic_1_LM_Negativity', 'qanda_topic_2_LM_Positivity', 'qanda_topic_2_LM_Negativity', 'qanda_topic_3_LM_Positivity', 'qanda_topic_3_LM_Negativity', 'qanda_topic_4_LM_Positivity', 'qanda_topic_4_LM_Negativity', 'qanda_topic_5_LM_Positivity', 'qanda_topic_5_LM_Negativity', 'qanda_topic_6_LM_Positivity', 'qanda_topic_6_LM_Negativity', 'qanda_topic_7_LM_Positivity', 'qanda_topic_7_LM_Negativity', 'qanda_topic_8_LM_Positivity', 'qanda_topic_8_LM_Negativity', 'qanda_topic_9_LM_Positivity', 'qanda_topic_9_LM_Negativity', 'qanda_topic_10_LM_Positivity', 'qanda_topic_10_LM_Negativity', 'qanda_topic_11_LM_Positivity', 'qanda_topic_11_LM_Negativity', 'qanda_topic_12_LM_Positivity', 'qanda_topic_12_LM_Negativity', 'qanda_topic_13_LM_Positivity', 'qanda_topic_13_LM_Negativity', 'qanda_topic_14_LM_Positivity', 'qanda_topic_14_LM_Negativity', 'qanda_topic_14_LM_Uncertainty']
#
#ADD_FEATURES = list(itertools.product([[], EA_FEATURES], 
#                                      [[], ALL_TS, OLEG_TS, MINI_TS, 
#                                       LDA_INTRO, LDA_QANDA, LDA_INTRO + LDA_QANDA,
#                                       LDA_INTRO_PN, LDA_QANDA_PN, LDA_INTRO_PN + LDA_QANDA_PN]))
#ADD_FEATURES = list(map(lambda l: l[0] + l[1], ADD_FEATURES))




#ADD_FEATURES = list(itertools.product([[], POL_FEATURES], #[f + ':roll-z' for f in POL_FEATURES]
#                                      [[], DISPERSION_FEATURES],
#                                      [[], FORECAST_ERROR_FEATURES], 
#                                      [[], RATIO_FEATURES]#, TRAILING_RATIO_FEATURES]
#                                      ))

#ADD_FEATURES = list(filter(lambda x: len(x)>0, 
#                           map(lambda l: l[0] + l[1] + l[2] + l[3], ADD_FEATURES)))


ADD_FEATURES = list(itertools.product([[], FORECAST_ERROR_FEATURES], 
                                      [[], RATIO_FEATURES]#, TRAILING_RATIO_FEATURES]
                                      ))

ADD_FEATURES = list(filter(lambda x: len(x)>0, 
                           map(lambda l: l[0] + l[1], ADD_FEATURES)))



es_jobs = [{
       'model': EarningsSurpriseModel('EP_surprise'),
       'features': ['EP_surprise'],
        'train_target': t,
        'return_target': t,
       } for t in TARGETS]

rs_jobs = [{
       'model': EarningsSurpriseModel('EP_ratio'),
       'features': ['EP_ratio'],
        'train_target': t,
        'return_target': t,
       } for t in TARGETS]


MODELS = [
        LogisticRegressionCV(solver = "lbfgs", max_iter = 1000, 
                             cv = 5, Cs = 100, n_jobs = -1),
        #Pipeline([('scaler', QuantileTransformer()), 
        #          ('LR', LogisticRegressionCV(solver = "lbfgs", max_iter = 1000, cv = 5, Cs = 100, n_jobs = -1))]),
        RandomForestClassifier(n_estimators = 5000, max_depth = 30, 
                               class_weight = "balanced_subsample", 
                               random_state = 0, n_jobs = -1)
        ]


ml_jobs = [{
       'model': m,
       'features': ef,
        'train_target': t,
        'return_target': t,
       }  for t in TARGETS for m in MODELS for ef in ADD_FEATURES]



#MODEL_JOBS = es_jobs + rs_jobs + ml_jobs
MODEL_JOBS = ml_jobs

MODEL_JOBS = [{**j, **GENERAL_MODEL_CONFIG} for j in MODEL_JOBS]


GENERAL_BACKTEST_CONFIG = {
            'default_commission': 0.001,
            'market_segment_commission': {'SP500TR': 0.001, 'SP400TR': 0.0015, 'SP600TR': 0.002},
            'market_commission': 0.0001,
            'start': pd.Timestamp('2013-01-01', tz = 'America/New_York'),
            'end': pd.Timestamp('2018-12-31', tz = 'America/New_York')
        }

"""
BACKTEST_JOBS = [
#        {
#        'name': 'buy-and-hold',
#        'strategy': buy_and_hold_strategy,
#        'strategy_args': {
#                }
#        },
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

"""

def generate_model_jobs():
    return MODEL_JOBS

#def generate_backtest_jobs():
#    return BACKTEST_JOBS

