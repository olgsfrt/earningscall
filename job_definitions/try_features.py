#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:50:02 2019

@author: mschnaubelt
"""

import pandas as pd
from backtest.strategy import call_strategy

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV

from util.validator import PhysicalTimeForwardValidation
import copy


# =============================================================================
model = RandomForestClassifier(n_estimators = 1000, max_depth = 5, #min_samples_split = 0.01,
                               class_weight = "balanced_subsample", 
                               random_state = 0, n_jobs = -1)
# =============================================================================



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
                    #'pays_dividend', 
                    'revenue_surprise', 
                    'revenue_surprise_std', 
                    'revenue_surprise_estimates', 
                    #'same_day_call_count', 
                    #'hour_of_day_half', 
                    'log_length', 
                    'nr_analysts', #'nr_executives',
                    'general_PositivityLM', 'general_NegativityLM', 
                    'qanda_PositivityLM', 'qanda_NegativityLM', 
                    'dps_ratio', 'dps_surprise', 'dps_surprise_std',
                    'cfps_surprise', 'cfps_ratio', 'cfps_eps_diff',
                    'book_market_ratio', #'rps' (revenue per share)
                    
                    #'nr_analyst_questions',
                    #'nr_executive_answers',
                    #'General_LM_Positivity','General_LM_Negativity',
                    #'LM_Positivity_Executives','LM_Negativity_Executives',
                    #'LM_Positivity_Analysts','LM_Negativity_Analysts',
                    
                    #'General_LM_Sentiment','General_LM_Uncertainty',
                    #'LM_Sentiment_Analysts', 
                    #'LM_Sentiment_Executives','LM_Uncertainty_Executives',
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
                    ], #+ ['%d'%c for c in range(30)],
        'top_flop_cutoff': 0.1,
        'validator': PhysicalTimeForwardValidation('2013-01-01', pd.Timedelta(12, 'M'), 
                                                   1500, 'final_datetime'),
        'rolling_window_size': 1500,
        'calculate_permutation_feature_importances': False,
        'calculate_partial_dependence': True
       }


FEATURE_TRIALS = [
        [],
        #['LM_Sentiment_Analysts','LM_Positivity_Analysts','LM_Negativity_Analysts','LM_Uncertainty_Analysts','LM_Sentiment_Executives','LM_Positivity_Executives','LM_Negativity_Executives','LM_Uncertainty_Executives'],
        #['length_general','length_qanda'],
        #['LM_Sentiment_Executives','LM_Positivity_Executives','LM_Negativity_Executives','LM_Uncertainty_Executives'],
        #['LM_Sentiment_Analysts','LM_Positivity_Analysts','LM_Negativity_Analysts','LM_Uncertainty_Analysts'],
        #['LM_Positivity_Analysts','LM_Negativity_Analysts','LM_Positivity_Executives','LM_Negativity_Executives'],

        #['assets','assets_LM_Sentiment','assets_LM_Positivity','assets_LM_Negativity','assets_LM_Uncertainty','liabilities','liabilities_LM_Sentiment','liabilities_LM_Positivity','liabilities_LM_Negativity','liabilities_LM_Uncertainty','equity','equity_LM_Sentiment','equity_LM_Positivity','equity_LM_Negativity','equity_LM_Uncertainty','revenues','revenues_LM_Sentiment','revenues_LM_Positivity','revenues_LM_Negativity','revenues_LM_Uncertainty','expenses','expenses_LM_Sentiment','expenses_LM_Positivity','expenses_LM_Negativity','expenses_LM_Uncertainty','loss','loss_LM_Sentiment','loss_LM_Positivity','loss_LM_Negativity','loss_LM_Uncertainty','earnings','earnings_LM_Sentiment','earnings_LM_Positivity','earnings_LM_Negativity','earnings_LM_Uncertainty','mergers','mergers_LM_Sentiment','mergers_LM_Positivity','mergers_LM_Negativity','mergers_LM_Uncertainty','strategy','strategy_LM_Sentiment','strategy_LM_Positivity','strategy_LM_Negativity','strategy_LM_Uncertainty','innovation','innovation_LM_Sentiment','innovation_LM_Positivity','innovation_LM_Negativity','innovation_LM_Uncertainty','f_markets','f_markets_LM_Sentiment','f_markets_LM_Positivity','f_markets_LM_Negativity','f_markets_LM_Uncertainty','distribution','distribution_LM_Sentiment','distribution_LM_Positivity','distribution_LM_Negativity','distribution_LM_Uncertainty','outlook','outlook_LM_Sentiment','outlook_LM_Positivity','outlook_LM_Negativity','outlook_LM_Uncertainty','environment','environment_LM_Sentiment','environment_LM_Positivity','environment_LM_Negativity','environment_LM_Uncertainty','liquidity','liquidity_LM_Sentiment','liquidity_LM_Positivity','liquidity_LM_Negativity','liquidity_LM_Uncertainty','movements','movements_LM_Sentiment','movements_LM_Positivity','movements_LM_Negativity','movements_LM_Uncertainty'],
        #['assets_q','assets_q_LM_Sentiment','assets_q_LM_Positivity','assets_q_LM_Negativity','assets_q_LM_Uncertainty','liabilities_q','liabilities_q_LM_Sentiment','liabilities_q_LM_Positivity','liabilities_q_LM_Negativity','liabilities_q_LM_Uncertainty','equity_q','equity_q_LM_Sentiment','equity_q_LM_Positivity','equity_q_LM_Negativity','equity_q_LM_Uncertainty','revenues_q','revenues_q_LM_Sentiment','revenues_q_LM_Positivity','revenues_q_LM_Negativity','revenues_q_LM_Uncertainty','expenses_q','expenses_q_LM_Sentiment','expenses_q_LM_Positivity','expenses_q_LM_Negativity','expenses_q_LM_Uncertainty','loss_q','loss_q_LM_Sentiment','loss_q_LM_Positivity','loss_q_LM_Negativity','loss_q_LM_Uncertainty','earnings_q','earnings_q_LM_Sentiment','earnings_q_LM_Positivity','earnings_q_LM_Negativity','earnings_q_LM_Uncertainty','mergers_q','mergers_q_LM_Sentiment','mergers_q_LM_Positivity','mergers_q_LM_Negativity','mergers_q_LM_Uncertainty','strategy_q','strategy_q_LM_Sentiment','strategy_q_LM_Positivity','strategy_q_LM_Negativity','strategy_q_LM_Uncertainty','innovation_q','innovation_q_LM_Sentiment','innovation_q_LM_Positivity','innovation_q_LM_Negativity','innovation_q_LM_Uncertainty','f_markets_q','f_markets_q_LM_Sentiment','f_markets_q_LM_Positivity','f_markets_q_LM_Negativity','f_markets_q_LM_Uncertainty','distribution_q','distribution_q_LM_Sentiment','distribution_q_LM_Positivity','distribution_q_LM_Negativity','distribution_q_LM_Uncertainty','outlook_q','outlook_q_LM_Sentiment','outlook_q_LM_Positivity','outlook_q_LM_Negativity','outlook_q_LM_Uncertainty','environment_q','environment_q_LM_Sentiment','environment_q_LM_Positivity','environment_q_LM_Negativity','environment_q_LM_Uncertainty','liquidity_q','liquidity_q_LM_Sentiment','liquidity_q_LM_Positivity','liquidity_q_LM_Negativity','liquidity_q_LM_Uncertainty','movements_q','movements_q_LM_Sentiment','movements_q_LM_Positivity','movements_q_LM_Negativity','movements_q_LM_Uncertainty'],
        #['assets_s','assets_s_LM_Sentiment','assets_s_LM_Positivity','assets_s_LM_Negativity','assets_s_LM_Uncertainty','liabilities_s','liabilities_s_LM_Sentiment','liabilities_s_LM_Positivity','liabilities_s_LM_Negativity','liabilities_s_LM_Uncertainty','equity_s','equity_s_LM_Sentiment','equity_s_LM_Positivity','equity_s_LM_Negativity','equity_s_LM_Uncertainty','revenues_s','revenues_s_LM_Sentiment','revenues_s_LM_Positivity','revenues_s_LM_Negativity','revenues_s_LM_Uncertainty','expenses_s','expenses_s_LM_Sentiment','expenses_s_LM_Positivity','expenses_s_LM_Negativity','expenses_s_LM_Uncertainty','loss_s','loss_s_LM_Sentiment','loss_s_LM_Positivity','loss_s_LM_Negativity','loss_s_LM_Uncertainty','earnings_s','earnings_s_LM_Sentiment','earnings_s_LM_Positivity','earnings_s_LM_Negativity','earnings_s_LM_Uncertainty','mergers_s','mergers_s_LM_Sentiment','mergers_s_LM_Positivity','mergers_s_LM_Negativity','mergers_s_LM_Uncertainty','strategy_s','strategy_s_LM_Sentiment','strategy_s_LM_Positivity','strategy_s_LM_Negativity','strategy_s_LM_Uncertainty','innovation_s','innovation_s_LM_Sentiment','innovation_s_LM_Positivity','innovation_s_LM_Negativity','innovation_s_LM_Uncertainty','f_markets_s','f_markets_s_LM_Sentiment','f_markets_s_LM_Positivity','f_markets_s_LM_Negativity','f_markets_s_LM_Uncertainty','distribution_s','distribution_s_LM_Sentiment','distribution_s_LM_Positivity','distribution_s_LM_Negativity','distribution_s_LM_Uncertainty','outlook_s','outlook_s_LM_Sentiment','outlook_s_LM_Positivity','outlook_s_LM_Negativity','outlook_s_LM_Uncertainty','environment_s','environment_s_LM_Sentiment','environment_s_LM_Positivity','environment_s_LM_Negativity','environment_s_LM_Uncertainty','liquidity_s','liquidity_s_LM_Sentiment','liquidity_s_LM_Positivity','liquidity_s_LM_Negativity','liquidity_s_LM_Uncertainty','movements_s','movements_s_LM_Sentiment','movements_s_LM_Positivity','movements_s_LM_Negativity','movements_s_LM_Uncertainty'],
        #['assets_q_s','assets_q_s_LM_Sentiment','assets_q_s_LM_Positivity','assets_q_s_LM_Negativity','assets_q_s_LM_Uncertainty','liabilities_q_s','liabilities_q_s_LM_Sentiment','liabilities_q_s_LM_Positivity','liabilities_q_s_LM_Negativity','liabilities_q_s_LM_Uncertainty','equity_q_s','equity_q_s_LM_Sentiment','equity_q_s_LM_Positivity','equity_q_s_LM_Negativity','equity_q_s_LM_Uncertainty','revenues_q_s','revenues_q_s_LM_Sentiment','revenues_q_s_LM_Positivity','revenues_q_s_LM_Negativity','revenues_q_s_LM_Uncertainty','expenses_q_s','expenses_q_s_LM_Sentiment','expenses_q_s_LM_Positivity','expenses_q_s_LM_Negativity','expenses_q_s_LM_Uncertainty','loss_q_s','loss_q_s_LM_Sentiment','loss_q_s_LM_Positivity','loss_q_s_LM_Negativity','loss_q_s_LM_Uncertainty','earnings_q_s','earnings_q_s_LM_Sentiment','earnings_q_s_LM_Positivity','earnings_q_s_LM_Negativity','earnings_q_s_LM_Uncertainty','mergers_q_s','mergers_q_s_LM_Sentiment','mergers_q_s_LM_Positivity','mergers_q_s_LM_Negativity','mergers_q_s_LM_Uncertainty','strategy_q_s','strategy_q_s_LM_Sentiment', 'strategy_q_s_LM_Positivity','strategy_q_s_LM_Negativity','strategy_q_s_LM_Uncertainty','innovation_q_s','innovation_q_s_LM_Sentiment','innovation_q_s_LM_Positivity','innovation_q_s_LM_Negativity','innovation_q_s_LM_Uncertainty','f_markets_q_s','f_markets_q_s_LM_Sentiment','f_markets_q_s_LM_Positivity','f_markets_q_s_LM_Negativity','f_markets_q_s_LM_Uncertainty','distribution_q_s','distribution_q_s_LM_Sentiment','distribution_q_s_LM_Positivity','distribution_q_s_LM_Negativity','distribution_q_s_LM_Uncertainty','outlook_q_s','outlook_q_s_LM_Sentiment','outlook_q_s_LM_Positivity','outlook_q_s_LM_Negativity','outlook_q_s_LM_Uncertainty','environment_q_s','environment_q_s_LM_Sentiment','environment_q_s_LM_Positivity','environment_q_s_LM_Negativity','environment_q_s_LM_Uncertainty','liquidity_q_s','liquidity_q_s_LM_Sentiment','liquidity_q_s_LM_Positivity','liquidity_q_s_LM_Negativity','liquidity_q_s_LM_Uncertainty','movements_q_s','movements_q_s_LM_Sentiment','movements_q_s_LM_Positivity','movements_q_s_LM_Negativity','movements_q_s_LM_Uncertainty'],
        
        #['assets','assets_LM_Positivity','assets_LM_Negativity','liabilities','liabilities_LM_Positivity','liabilities_LM_Negativity','equity','equity_LM_Positivity','equity_LM_Negativity','revenues','revenues_LM_Positivity','revenues_LM_Negativity','expenses','expenses_LM_Positivity','expenses_LM_Negativity','loss','loss_LM_Positivity','loss_LM_Negativity','earnings','earnings_LM_Positivity','earnings_LM_Negativity','mergers','mergers_LM_Positivity','mergers_LM_Negativity','strategy','strategy_LM_Positivity','strategy_LM_Negativity','innovation','innovation_LM_Positivity','innovation_LM_Negativity','f_markets','f_markets_LM_Positivity','f_markets_LM_Negativity','distribution','distribution_LM_Positivity','distribution_LM_Negativity','outlook','outlook_LM_Positivity','outlook_LM_Negativity','environment','environment_LM_Positivity','environment_LM_Negativity','liquidity','liquidity_LM_Positivity','liquidity_LM_Negativity','movements','movements_LM_Positivity','movements_LM_Negativity','movements_LM_Uncertainty'],
        #['assets_q','assets_q_LM_Positivity','assets_q_LM_Negativity','liabilities_q','liabilities_q_LM_Positivity','liabilities_q_LM_Negativity','equity_q','equity_q_LM_Positivity','equity_q_LM_Negativity','revenues_q','revenues_q_LM_Positivity','revenues_q_LM_Negativity','expenses_q','expenses_q_LM_Positivity','expenses_q_LM_Negativity','loss_q','loss_q_LM_Positivity','loss_q_LM_Negativity','earnings_q','earnings_q_LM_Positivity','earnings_q_LM_Negativity','mergers_q','mergers_q_LM_Positivity','mergers_q_LM_Negativity','strategy_q','strategy_q_LM_Positivity','strategy_q_LM_Negativity','innovation_q','innovation_q_LM_Positivity','innovation_q_LM_Negativity','f_markets_q','f_markets_q_LM_Positivity','f_markets_q_LM_Negativity','distribution_q','distribution_q_LM_Positivity','distribution_q_LM_Negativity','outlook_q','outlook_q_LM_Positivity','outlook_q_LM_Negativity','environment_q','environment_q_LM_Positivity','environment_q_LM_Negativity','liquidity_q','liquidity_q_LM_Positivity','liquidity_q_LM_Negativity','movements_q','movements_q_LM_Positivity','movements_q_LM_Negativity','movements_q_LM_Uncertainty'],
        #['assets_s','assets_s_LM_Positivity','assets_s_LM_Negativity','liabilities_s','liabilities_s_LM_Positivity','liabilities_s_LM_Negativity','equity_s','equity_s_LM_Positivity','equity_s_LM_Negativity','revenues_s','revenues_s_LM_Positivity','revenues_s_LM_Negativity','expenses_s','expenses_s_LM_Positivity','expenses_s_LM_Negativity','loss_s','loss_s_LM_Positivity','loss_s_LM_Negativity','earnings_s','earnings_s_LM_Positivity','earnings_s_LM_Negativity','mergers_s','mergers_s_LM_Positivity','mergers_s_LM_Negativity','strategy_s','strategy_s_LM_Positivity','strategy_s_LM_Negativity','innovation_s','innovation_s_LM_Positivity','innovation_s_LM_Negativity','f_markets_s','f_markets_s_LM_Positivity','f_markets_s_LM_Negativity','distribution_s','distribution_s_LM_Positivity','distribution_s_LM_Negativity','outlook_s','outlook_s_LM_Positivity','outlook_s_LM_Negativity','environment_s','environment_s_LM_Positivity','environment_s_LM_Negativity','liquidity_s','liquidity_s_LM_Positivity','liquidity_s_LM_Negativity','movements_s','movements_s_LM_Positivity','movements_s_LM_Negativity','movements_s_LM_Uncertainty'],
        #['assets_q_s','assets_q_s_LM_Positivity','assets_q_s_LM_Negativity','liabilities_q_s','liabilities_q_s_LM_Positivity','liabilities_q_s_LM_Negativity','equity_q_s','equity_q_s_LM_Positivity','equity_q_s_LM_Negativity','revenues_q_s','revenues_q_s_LM_Positivity','revenues_q_s_LM_Negativity','expenses_q_s','expenses_q_s_LM_Positivity','expenses_q_s_LM_Negativity','loss_q_s','loss_q_s_LM_Positivity','loss_q_s_LM_Negativity','earnings_q_s','earnings_q_s_LM_Positivity','earnings_q_s_LM_Negativity','mergers_q_s','mergers_q_s_LM_Positivity','mergers_q_s_LM_Negativity','strategy_q_s', 'strategy_q_s_LM_Positivity','strategy_q_s_LM_Negativity','innovation_q_s','innovation_q_s_LM_Positivity','innovation_q_s_LM_Negativity','f_markets_q_s','f_markets_q_s_LM_Positivity','f_markets_q_s_LM_Negativity','distribution_q_s','distribution_q_s_LM_Positivity','distribution_q_s_LM_Negativity','outlook_q_s','outlook_q_s_LM_Positivity','outlook_q_s_LM_Negativity','environment_q_s','environment_q_s_LM_Positivity','environment_q_s_LM_Negativity','liquidity_q_s','liquidity_q_s_LM_Positivity','liquidity_q_s_LM_Negativity','movements_q_s','movements_q_s_LM_Positivity','movements_q_s_LM_Negativity','movements_q_s_LM_Uncertainty'],
        
        #['strategy_q_s_LM_Positivity','strategy_q_s_LM_Negativity','outlook_q_s_LM_Positivity','outlook_q_s_LM_Negativity','environment_q_s_LM_Positivity','environment_q_s_LM_Negativity','liquidity_q_s_LM_Positivity','liquidity_q_s_LM_Negativity'],
        #['outlook_q_s_LM_Positivity','outlook_q_s_LM_Negativity','environment_q_s_LM_Positivity','environment_q_s_LM_Negativity'],
        #['assets_LM_Positivity','assets_LM_Negativity','liabilities_LM_Positivity','liabilities_LM_Negativity','equity_LM_Positivity','equity_LM_Negativity','revenues_LM_Positivity','revenues_LM_Negativity','expenses_LM_Positivity','expenses_LM_Negativity','loss_LM_Positivity','loss_LM_Negativity','earnings_LM_Positivity','earnings_LM_Negativity','mergers_LM_Positivity','mergers_LM_Negativity','strategy_LM_Positivity','strategy_LM_Negativity','innovation_LM_Positivity','innovation_LM_Negativity','f_markets_LM_Positivity','f_markets_LM_Negativity','distribution_LM_Positivity','distribution_LM_Negativity','outlook_LM_Positivity','outlook_LM_Negativity','environment_LM_Positivity','environment_LM_Negativity','liquidity_LM_Positivity','liquidity_LM_Negativity','movements_LM_Positivity','movements_LM_Negativity'],
        #['assets_q_LM_Positivity','assets_q_LM_Negativity','liabilities_q_LM_Positivity','liabilities_q_LM_Negativity','equity_q_LM_Positivity','equity_q_LM_Negativity','revenues_q_LM_Positivity','revenues_q_LM_Negativity','expenses_q_LM_Positivity','expenses_q_LM_Negativity','loss_q_LM_Positivity','loss_q_LM_Negativity','earnings_q_LM_Positivity','earnings_q_LM_Negativity','mergers_q_LM_Positivity','mergers_q_LM_Negativity','strategy_q_LM_Positivity','strategy_q_LM_Negativity','innovation_q_LM_Positivity','innovation_q_LM_Negativity','f_markets_q_LM_Positivity','f_markets_q_LM_Negativity','distribution_q_LM_Positivity','distribution_q_LM_Negativity','outlook_q_LM_Positivity','outlook_q_LM_Negativity','environment_q_LM_Positivity','environment_q_LM_Negativity','liquidity_q_LM_Positivity','liquidity_q_LM_Negativity','movements_q_LM_Positivity','movements_q_LM_Negativity'],
        #['assets_s_LM_Positivity','assets_s_LM_Negativity','liabilities_s_LM_Positivity','liabilities_s_LM_Negativity','equity_s_LM_Positivity','equity_s_LM_Negativity','revenues_s_LM_Positivity','revenues_s_LM_Negativity','expenses_s_LM_Positivity','expenses_s_LM_Negativity','loss_s_LM_Positivity','loss_s_LM_Negativity','earnings_s_LM_Positivity','earnings_s_LM_Negativity','mergers_s_LM_Positivity','mergers_s_LM_Negativity','strategy_s_LM_Positivity','strategy_s_LM_Negativity','innovation_s_LM_Positivity','innovation_s_LM_Negativity','f_markets_s_LM_Positivity','f_markets_s_LM_Negativity','distribution_s_LM_Positivity','distribution_s_LM_Negativity','outlook_s_LM_Positivity','outlook_s_LM_Negativity','environment_s_LM_Positivity','environment_s_LM_Negativity','liquidity_s_LM_Positivity','liquidity_s_LM_Negativity','movements_s_LM_Positivity','movements_s_LM_Negativity'],
        #['assets_q_s_LM_Positivity','assets_q_s_LM_Negativity','liabilities_q_s_LM_Positivity','liabilities_q_s_LM_Negativity','equity_q_s_LM_Positivity','equity_q_s_LM_Negativity','revenues_q_s_LM_Positivity','revenues_q_s_LM_Negativity','expenses_q_s_LM_Positivity','expenses_q_s_LM_Negativity','loss_q_s_LM_Positivity','loss_q_s_LM_Negativity','earnings_q_s_LM_Positivity','earnings_q_s_LM_Negativity','mergers_q_s_LM_Positivity','mergers_q_s_LM_Negativity','strategy_q_s_LM_Positivity','strategy_q_s_LM_Negativity','innovation_q_s_LM_Positivity','innovation_q_s_LM_Negativity','f_markets_q_s_LM_Positivity','f_markets_q_s_LM_Negativity','distribution_q_s_LM_Positivity','distribution_q_s_LM_Negativity','outlook_q_s_LM_Positivity','outlook_q_s_LM_Negativity','environment_q_s_LM_Positivity','environment_q_s_LM_Negativity','liquidity_q_s_LM_Positivity','liquidity_q_s_LM_Negativity','movements_q_s_LM_Positivity','movements_q_s_LM_Negativity'],
        
        #['outlook_q_s_LM_Positivity','outlook_q_s_LM_Negativity','distribution_q_s_LM_Positivity','distribution_q_s_LM_Negativity','movements_q_s_LM_Positivity','movements_q_s_LM_Negativity'],
        #['outlook_s_LM_Positivity','outlook_s_LM_Negativity','distribution_s_LM_Positivity','distribution_s_LM_Negativity','movements_s_LM_Positivity','movements_s_LM_Negativity'],
        #['outlook_q_LM_Positivity','outlook_q_LM_Negativity','distribution_q_LM_Positivity','distribution_q_LM_Negativity','movements_q_LM_Positivity','movements_q_LM_Negativity'],
        #['outlook_LM_Positivity','outlook_LM_Negativity','distribution_LM_Positivity','distribution_LM_Negativity','movements_LM_Positivity','movements_LM_Negativity'],
        #['outlook_q_s_LM_Positivity','outlook_q_s_LM_Negativity','distribution_q_s_LM_Positivity','distribution_q_s_LM_Negativity','movements_q_s_LM_Positivity','movements_q_s_LM_Negativity','outlook_s_LM_Positivity','outlook_s_LM_Negativity','distribution_s_LM_Positivity','distribution_s_LM_Negativity','movements_s_LM_Positivity','movements_s_LM_Negativity'],
        #['outlook_q_s_LM_Positivity','outlook_q_s_LM_Negativity','environment_q_s_LM_Positivity','environment_q_s_LM_Negativity','outlook_s_LM_Positivity','outlook_s_LM_Negativity','environment_s_LM_Positivity','environment_s_LM_Negativity'],
        
        #['First_Ana_Sent','First_Ana_Pos','First_Ana_Neg','First_Ana_Unc','Last_Ana_Sent','Last_Ana_Pos','Last_Ana_Neg','Last_Ana_Unc'],
        ['LM_Sentiment_Executives','LM_Positivity_Executives','First_Ana_Pos','First_Ana_Neg','Last_Ana_Pos','Last_Ana_Neg'],
        
    ['assets_LM_Positivity','assets_LM_Negativity','revenues_LM_Positivity','revenues_LM_Negativity','expenses_LM_Positivity','expenses_LM_Negativity','earnings_LM_Positivity','earnings_LM_Negativity','innovation_LM_Positivity','innovation_LM_Negativity','outlook_LM_Positivity','outlook_LM_Negativity','environment_LM_Positivity','environment_LM_Negativity','movements_LM_Positivity','movements_LM_Negativity'],
    ['revenues_q_LM_Positivity', 'revenues_q_LM_Negativity', 'outlook_q_LM_Positivity', 'outlook_q_LM_Negativity', 'environment_q_LM_Positivity', 'environment_q_LM_Negativity', 'movements_q_LM_Positivity', 'movements_q_LM_Negativity'],
    ['assets_LM_Positivity', 'assets_LM_Negativity', 'revenues_LM_Positivity', 'revenues_LM_Negativity', 'expenses_LM_Positivity', 'expenses_LM_Negativity', 'earnings_LM_Positivity', 'earnings_LM_Negativity', 'innovation_LM_Positivity', 'innovation_LM_Negativity', 'outlook_LM_Positivity', 'outlook_LM_Negativity', 'environment_LM_Positivity', 'environment_LM_Negativity', 'movements_LM_Positivity', 'movements_LM_Negativity','revenues_q_LM_Positivity', 'revenues_q_LM_Negativity', 'outlook_q_LM_Positivity', 'outlook_q_LM_Negativity', 'environment_q_LM_Positivity', 'environment_q_LM_Negativity', 'movements_q_LM_Positivity', 'movements_q_LM_Negativity'],
        
        ['outlook_q_s_LM_Positivity','outlook_q_s_LM_Negativity','environment_q_s_LM_Positivity','environment_q_s_LM_Negativity'],
        ['outlook_s_LM_Positivity','outlook_s_LM_Negativity','environment_s_LM_Positivity','environment_s_LM_Negativity'],
        ['outlook_q_s_LM_Positivity','outlook_q_s_LM_Negativity','environment_q_s_LM_Positivity','environment_q_s_LM_Negativity','outlook_s_LM_Positivity','outlook_s_LM_Negativity','environment_s_LM_Positivity','environment_s_LM_Negativity'],
        ['outlook_q_s_LM_Positivity','outlook_q_s_LM_Negativity','distribution_q_s_LM_Positivity','distribution_q_s_LM_Negativity','movements_q_s_LM_Positivity','movements_q_s_LM_Negativity'],
        ['outlook_s_LM_Positivity','outlook_s_LM_Negativity','distribution_s_LM_Positivity','distribution_s_LM_Negativity','movements_s_LM_Positivity','movements_s_LM_Negativity'],
        
        ['outlook_q_s_LM_Positivity','outlook_q_s_LM_Negativity','distribution_q_s_LM_Positivity','distribution_q_s_LM_Negativity','movements_q_s_LM_Positivity','movements_q_s_LM_Negativity','outlook_s_LM_Positivity','outlook_s_LM_Negativity','distribution_s_LM_Positivity','distribution_s_LM_Negativity','movements_s_LM_Positivity','movements_s_LM_Negativity'],
    ]

JOBS = []
for ft in FEATURE_TRIALS:
    j = copy.deepcopy(job)
    j['features'] += ft
    JOBS.append(j)



def generate_model_jobs():
    return JOBS

from job_definitions.final import generate_backtest_jobs
