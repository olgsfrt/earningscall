#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:54:34 2019

@author: mschnaubelt
"""

CALL_FILE = 'E:/data/con_dict_small_test.json'
EXPORT_FOLDER = 'D:/01_Diss_Data/Earnings_Call/Experiments/00_test/EXPORT_FOLDER'
INDICES = {'SP500TR': 'index_0#.SPX.json', 'SP400TR': 'index_0#.SP400.json', 'SP600TR': 'index_0#.SPCY.json'}

TMP_FOLDER = 'D:/01_Diss_Data/Earnings_Call/Experiments/00_test/TMP_FOLDER'

RUNS_FOLDER = 'D:/01_Diss_Data/Earnings_Call/Experiments/00_test/RUNS_FOLDER'

#TARGETS = ['ff-dec_abnormal_-5d_drift', 'ff-dec_abnormal_5d_drift', 
#           'ff-dec_abnormal_20d_drift', 'ff-dec_abnormal_60d_drift']

TARGETS = ['ff5_abnormal_-5d_drift', 'ff5_abnormal_5d_drift', 
           'ff5_abnormal_20d_drift', 'ff5_abnormal_60d_drift']


MODEL_ORDER = ['ES-EP-surprise', 'LR', 'LR-Q', 'LASSO', 'LASSO-Q', 'RF']


RATIO_FEATURES = ['MV_log', 'BM_ratio', 'EP_ratio', 
                  'SP_ratio', 'CP_ratio', 
                  'DY_ratio', 'dividend_payout_ratio']

TRAILING_RATIO_FEATURES = ['MV_log', 'BM_trailing_ratio', 'EP_trailing_ratio', 
                           'SP_trailing_ratio', 'CP_trailing_ratio', 
                           'DY_trailing_ratio', 'dividend_payout_trailing_ratio']

FORECAST_ERROR_FEATURES = ['BM_surprise', 'EP_surprise', 'SP_surprise', 
                           'DY_surprise', 'CP_surprise']

DISPERSION_FEATURES = ['EP_surprise_mean_std', 'EP_surprise_std', 
                       'EP_surprise_revisions', 'EP_surprise_estimates', 
                       'SP_surprise_std', 'SP_surprise_estimates', 
                       'DY_surprise_std']

POL_FEATURES = []



#POL_FEATURES = ['general_PositivityLM','general_NegativityLM',
               # 'qanda_PositivityLM','qanda_NegativityLM',
               # 'First_Ana_Pos','First_Ana_Neg','Last_Ana_Pos','Last_Ana_Neg',
               # 'revenues_si_LM_Positivity','revenues_si_LM_Negativity',
               # 'earnings_si_LM_Positivity','earnings_si_LM_Negativity',
               # 'outlook_si_LM_Positivity','outlook_si_LM_Negativity',
               # 'environment_si_LM_Positivity','environment_si_LM_Negativity',
               # 'liquidity_si_LM_Positivity','liquidity_si_LM_Negativity',
               # 'movements_si_LM_Positivity','movements_si_LM_Negativity',
               # 'revenues_sq_LM_Positivity','revenues_sq_LM_Negativity',
               # 'earnings_sq_LM_Positivity','earnings_sq_LM_Negativity',
               # 'outlook_sq_LM_Positivity','outlook_sq_LM_Negativity',
               # 'environment_sq_LM_Positivity','environment_sq_LM_Negativity',
               # 'liquidity_sq_LM_Positivity','liquidity_sq_LM_Negativity',
               # 'movements_sq_LM_Positivity','movements_sq_LM_Negativity']
RPOL_FEATURES = [f + ':roll-z' for f in POL_FEATURES]


#SEN_POL_FEATURES = ['general_PositivityLM','general_NegativityLM','qanda_PositivityLM','qanda_NegativityLM','First_Ana_Pos','First_Ana_Neg','Last_Ana_Pos','Last_Ana_Neg','assets_si_LM_Positivity','assets_si_LM_Negativity','liabilities_si_LM_Positivity','liabilities_si_LM_Negativity','equity_si_LM_Positivity','equity_si_LM_Negativity','revenues_si_LM_Positivity','revenues_si_LM_Negativity','expenses_si_LM_Positivity','expenses_si_LM_Negativity','loss_si_LM_Positivity','loss_si_LM_Negativity','earnings_si_LM_Positivity','earnings_si_LM_Negativity','mergers_si_LM_Positivity','mergers_si_LM_Negativity','strategy_si_LM_Positivity','strategy_si_LM_Negativity','innovation_si_LM_Positivity','innovation_si_LM_Negativity','f_markets_si_LM_Positivity','f_markets_si_LM_Negativity','distribution_si_LM_Positivity','distribution_si_LM_Negativity','outlook_si_LM_Positivity','outlook_si_LM_Negativity','environment_si_LM_Positivity','environment_si_LM_Negativity','liquidity_si_LM_Positivity','liquidity_si_LM_Negativity','movements_si_LM_Positivity','movements_si_LM_Negativity','assets_sq_LM_Positivity','assets_sq_LM_Negativity','liabilities_sq_LM_Positivity','liabilities_sq_LM_Negativity','equity_sq_LM_Positivity','equity_sq_LM_Negativity','revenues_sq_LM_Positivity','revenues_sq_LM_Negativity','expenses_sq_LM_Positivity','expenses_sq_LM_Negativity','loss_sq_LM_Positivity','loss_sq_LM_Negativity','earnings_sq_LM_Positivity','earnings_sq_LM_Negativity','mergers_sq_LM_Positivity','mergers_sq_LM_Negativity','strategy_sq_LM_Positivity','strategy_sq_LM_Negativity','innovation_sq_LM_Positivity','innovation_sq_LM_Negativity','f_markets_sq_LM_Positivity','f_markets_sq_LM_Negativity','distribution_sq_LM_Positivity','distribution_sq_LM_Negativity','outlook_sq_LM_Positivity','outlook_sq_LM_Negativity','environment_sq_LM_Positivity','environment_sq_LM_Negativity','liquidity_sq_LM_Positivity','liquidity_sq_LM_Negativity','movements_sq_LM_Positivity','movements_sq_LM_Negativity']
#PAR_POL_FEATURES = ['general_PositivityLM','general_NegativityLM','qanda_PositivityLM','qanda_NegativityLM','First_Ana_Pos','First_Ana_Neg','Last_Ana_Pos','Last_Ana_Neg','assets_pi_LM_Positivity','assets_pi_LM_Negativity','liabilities_pi_LM_Positivity','liabilities_pi_LM_Negativity','equity_pi_LM_Positivity','equity_pi_LM_Negativity','revenues_pi_LM_Positivity','revenues_pi_LM_Negativity','expenses_pi_LM_Positivity','expenses_pi_LM_Negativity','loss_pi_LM_Positivity','loss_pi_LM_Negativity','earnings_pi_LM_Positivity','earnings_pi_LM_Negativity','mergers_pi_LM_Positivity','mergers_pi_LM_Negativity','strategy_pi_LM_Positivity','strategy_pi_LM_Negativity','innovation_pi_LM_Positivity','innovation_pi_LM_Negativity','f_markets_pi_LM_Positivity','f_markets_pi_LM_Negativity','distribution_pi_LM_Positivity','distribution_pi_LM_Negativity','outlook_pi_LM_Positivity','outlook_pi_LM_Negativity','environment_pi_LM_Positivity','environment_pi_LM_Negativity','liquidity_pi_LM_Positivity','liquidity_pi_LM_Negativity','movements_pi_LM_Positivity','movements_pi_LM_Negativity','assets_pq_LM_Positivity','assets_pq_LM_Negativity','liabilities_pq_LM_Positivity','liabilities_pq_LM_Negativity','equity_pq_LM_Positivity','equity_pq_LM_Negativity','revenues_pq_LM_Positivity','revenues_pq_LM_Negativity','expenses_pq_LM_Positivity','expenses_pq_LM_Negativity','loss_pq_LM_Positivity','loss_pq_LM_Negativity','earnings_pq_LM_Positivity','earnings_pq_LM_Negativity','mergers_pq_LM_Positivity','mergers_pq_LM_Negativity','strategy_pq_LM_Positivity','strategy_pq_LM_Negativity','innovation_pq_LM_Positivity','innovation_pq_LM_Negativity','f_markets_pq_LM_Positivity','f_markets_pq_LM_Negativity','distribution_pq_LM_Positivity','distribution_pq_LM_Negativity','outlook_pq_LM_Positivity','outlook_pq_LM_Negativity','environment_pq_LM_Positivity','environment_pq_LM_Negativity','liquidity_pq_LM_Positivity','liquidity_pq_LM_Negativity','movements_pq_LM_Positivity','movements_pq_LM_Negativity']

BASE_FEATURES = RATIO_FEATURES + FORECAST_ERROR_FEATURES# + DISPERSION_FEATURES# + POL_FEATURES


FEATURE_SETS_ORDER = ['VR', 'TFR', 'FE', 'UIQ', 'POL', 'RPOL']

FEATURE_SETS_DICT = {
        'FE': FORECAST_ERROR_FEATURES,
        'VR': RATIO_FEATURES,
        'TFR': TRAILING_RATIO_FEATURES,
        'UIQ': DISPERSION_FEATURES,
        'POL': POL_FEATURES,
        'RPOL': RPOL_FEATURES
        }


FEATURE_NAME_DICT = {
        'MV_log': '$MV_{s,q}$', 
        'BM_ratio': '$BM_{s,q}$', 
        'EP_ratio': '$EP_{s,q}$', 
        'SP_ratio': '$SP_{s,q}$', 
        'CP_ratio': '$CP_{s,q}$', 
        'DY_ratio': '$DY_{s,q}$', 
        'dividend_payout_ratio': '$DR_{s,q}$', 
        'BM_surprise': '$BMFE_{s,q}$', 
        'EP_surprise': '$EPFE_{s,q}$', 
        'SP_surprise': '$SPFE_{s,q}$', 
        'DY_surprise': '$DYFE_{s,q}$', 
        'CP_surprise': '$CPFE_{s,q}$', 
        'EP_surprise_mean_std': '$V$-$EPS_{s,q}$', 
        'EP_surprise_std': '$D$-$EPS_{s,q}$', 
        'EP_surprise_revisions': '$C$-$EPS_{s,q}$', 
        'EP_surprise_estimates': '$N$-$EPS_{s,q}$', 
        'SP_surprise_std': '$D$-$REV_{s,q}$', 
        'SP_surprise_estimates': '$N$-$REV_{s,q}$', 
        'DY_surprise_std': '$D$-$DIV_{s,q}$', 
        'log_length_intro': '$I$-$LEN_{s,q}$', 
        'log_length_qanda': '$Q$-$LEN_{s,q}$', 
        'nr_analysts': '$N$-$ANA_{s,q}$', 
        'general_PositivityLM':'$I$-$P_{s,q}$', 
        'general_NegativityLM':'$I$-$N_{s,q}$',
        'qanda_PositivityLM':'$Q$-$P_{s,q}$', 
        'qanda_NegativityLM':'$Q$-$N_{s,q}$',
        'First_Ana_Pos':'$FA$-$P_{s,q}$', 
        'First_Ana_Neg':'$FA$-$N_{s,q}$', 
        'Last_Ana_Pos':'$SA$-$P_{s,q}$', 
        'Last_Ana_Neg':'$SA$-$N_{s,q}$', 
        'movements_si_LM_Positivity':'$I$-$CH$-$P_{s,q}$', 
        'movements_si_LM_Negativity':'$I$-$CH$-$N_{s,q}$', 
        'movements_sq_LM_Positivity':'$Q$-$CH$-$P_{s,q}$', 
        'movements_sq_LM_Negativity':'$Q$-$CH$-$N_{s,q}$', 
        'earnings_si_LM_Positivity':'$I$-$EA$-$P_{s,q}$', 
        'earnings_si_LM_Negativity':'$I$-$EA$-$N_{s,q}$', 
        'earnings_sq_LM_Positivity':'$Q$-$EA$-$P_{s,q}$', 
        'earnings_sq_LM_Negativity':'$Q$-$EA$-$N_{s,q}$', 
        'environment_si_LM_Positivity':'$I$-$EN$-$P_{s,q}$', 
        'environment_si_LM_Negativity':'$I$-$EN$-$N_{s,q}$', 
        'environment_sq_LM_Positivity':'$Q$-$EN$-$P_{s,q}$', 
        'environment_sq_LM_Negativity':'$Q$-$EN$-$N_{s,q}$', 
        'revenues_si_LM_Positivity':'$I$-$RE$-$P_{s,q}$', 
        'revenues_si_LM_Negativity':'$I$-$RE$-$N_{s,q}$', 
        'revenues_sq_LM_Positivity':'$Q$-$RE$-$P_{s,q}$', 
        'revenues_sq_LM_Negativity':'$Q$-$RE$-$N_{s,q}$', 
        'outlook_si_LM_Positivity':'$I$-$OU$-$P_{s,q}$', 
        'outlook_si_LM_Negativity':'$I$-$OU$-$N_{s,q}$', 
        'outlook_sq_LM_Positivity':'$Q$-$OU$-$P_{s,q}$', 
        'outlook_sq_LM_Negativity':'$Q$-$OU$-$N_{s,q}$', 
        'liquidity_si_LM_Positivity':'$I$-$LI$-$P_{s,q}$', 
        'liquidity_si_LM_Negativity':'$I$-$LI$-$N_{s,q}$', 
        'liquidity_sq_LM_Positivity':'$Q$-$LI$-$P_{s,q}$', 
        'liquidity_sq_LM_Negativity':'$Q$-$LI$-$N_{s,q}$',
        
        'general_PositivityLM:roll-z':'$I$-$P_{s,q}$', 
        'general_NegativityLM:roll-z':'$I$-$N_{s,q}$',
        'qanda_PositivityLM:roll-z':'$Q$-$P_{s,q}$', 
        'qanda_NegativityLM:roll-z':'$Q$-$N_{s,q}$',
        'First_Ana_Pos:roll-z':'$FA$-$P_{s,q}$', 
        'First_Ana_Neg:roll-z':'$FA$-$N_{s,q}$', 
        'Last_Ana_Pos:roll-z':'$SA$-$P_{s,q}$', 
        'Last_Ana_Neg:roll-z':'$SA$-$N_{s,q}$', 
        'movements_si_LM_Positivity:roll-z':'$I$-$CH$-$P_{s,q}$', 
        'movements_si_LM_Negativity:roll-z':'$I$-$CH$-$N_{s,q}$', 
        'movements_sq_LM_Positivity:roll-z':'$Q$-$CH$-$P_{s,q}$', 
        'movements_sq_LM_Negativity:roll-z':'$Q$-$CH$-$N_{s,q}$', 
        'earnings_si_LM_Positivity:roll-z':'$I$-$EA$-$P_{s,q}$', 
        'earnings_si_LM_Negativity:roll-z':'$I$-$EA$-$N_{s,q}$', 
        'earnings_sq_LM_Positivity:roll-z':'$Q$-$EA$-$P_{s,q}$', 
        'earnings_sq_LM_Negativity:roll-z':'$Q$-$EA$-$N_{s,q}$', 
        'environment_si_LM_Positivity:roll-z':'$I$-$EN$-$P_{s,q}$', 
        'environment_si_LM_Negativity:roll-z':'$I$-$EN$-$N_{s,q}$', 
        'environment_sq_LM_Positivity:roll-z':'$Q$-$EN$-$P_{s,q}$', 
        'environment_sq_LM_Negativity:roll-z':'$Q$-$EN$-$N_{s,q}$', 
        'revenues_si_LM_Positivity:roll-z':'$I$-$RE$-$P_{s,q}$', 
        'revenues_si_LM_Negativity:roll-z':'$I$-$RE$-$N_{s,q}$', 
        'revenues_sq_LM_Positivity:roll-z':'$Q$-$RE$-$P_{s,q}$', 
        'revenues_sq_LM_Negativity:roll-z':'$Q$-$RE$-$N_{s,q}$', 
        'outlook_si_LM_Positivity:roll-z':'$I$-$OU$-$P_{s,q}$', 
        'outlook_si_LM_Negativity:roll-z':'$I$-$OU$-$N_{s,q}$', 
        'outlook_sq_LM_Positivity:roll-z':'$Q$-$OU$-$P_{s,q}$', 
        'outlook_sq_LM_Negativity:roll-z':'$Q$-$OU$-$N_{s,q}$', 
        'liquidity_si_LM_Positivity:roll-z':'$I$-$LI$-$P_{s,q}$', 
        'liquidity_si_LM_Negativity:roll-z':'$I$-$LI$-$N_{s,q}$', 
        'liquidity_sq_LM_Positivity:roll-z':'$Q$-$LI$-$P_{s,q}$', 
        'liquidity_sq_LM_Negativity:roll-z':'$Q$-$LI$-$N_{s,q}$',
        
        'ff-dec_abnormal_-5d_drift': '$Y^{-5}_{s,q}$',
        'ff-dec_abnormal_5d_drift': '$Y^{5}_{s,q}$',
        'ff-dec_abnormal_20d_drift': '$Y^{20}_{s,q}$',
        'ff-dec_abnormal_60d_drift': '$Y^{60}_{s,q}$',
        }



FINAL_RUN_FOLDERS = [
               'D:/01_Diss_Data/Earnings_Call/Experiments/00_test/FINAL_RUN_FOLDERS'
               ]

FINAL_TIME_DEP_RUN = 'D:/01_Diss_Data/Earnings_Call/Experiments/00_test/FINAL_TIME_DEP_RUN'

FINAL_ALE_RUN = 'D:/01_Diss_Data/Earnings_Call/Experiments/00_test/FINAL_ALE_RUN'
FINAL_ALE_2D_RUN = 'D:/01_Diss_Data/Earnings_Call/Experiments/00_test/FINAL_ALE_2D_RUN'

FINAL_BACKTEST_RUN = 'D:/01_Diss_Data/Earnings_Call/Experiments/00_test/FINAL_BACKTEST_RUN'

FINAL_SEED_RUN = 'D:/01_Diss_Data/Earnings_Call/Experiments/00_test/FINAL_SEED_RUN'
FINAL_PARAMETER_RUN = 'D:/01_Diss_Data/Earnings_Call/Experiments/00_test/FINAL_PARAMETER_RUN'
