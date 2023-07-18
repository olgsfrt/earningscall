#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:44:15 2019

@author: mschnaubelt
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import pyfolio as pf

from config import FINAL_BACKTEST_RUN


RUN_FOLDER = FINAL_BACKTEST_RUN
BACKTEST_NAME = '/backtest-0-per-day-5d-0.1-LS/'


JOBS = [
        ('1ES', 'job-0', 'backtest-1-per-day-5d-0.1-LS', 'backtest-2-per-day-5d-0.1-LS-noTAC'), 
        ('2LR', 'job-1', 'backtest-1-per-day-5d-0.1-LS', 'backtest-2-per-day-5d-0.1-LS-noTAC'), 
        ('3LR-B', 'job-2', 'backtest-1-per-day-5d-0.1-LS', 'backtest-2-per-day-5d-0.1-LS-noTAC'), 
        ('4RF', 'job-3', 'backtest-1-per-day-5d-0.1-LS', 'backtest-2-per-day-5d-0.1-LS-noTAC'), 
        ]


def read_jobs(run_folder, jobs):
    for model_name, job, backtest_job, backtest_job_noTAC in jobs:
        if not os.path.isfile('%s/%s/%s/stats.hdf' % (run_folder, job, backtest_job)):
            continue
        result = pd.read_hdf('%s/%s/model_results.hdf' % (run_folder, job))
        bt_result = pd.read_hdf('%s/%s/%s/results.hdf' % (run_folder, job, backtest_job))
        
        bt_stats = pd.read_hdf('%s/%s/%s/stats.hdf' % (run_folder, job, backtest_job))
        bt_stats_noTAC = pd.read_hdf('%s/%s/%s/stats.hdf' % (run_folder, job, backtest_job_noTAC))
        
        yield (model_name, job, result, bt_result, bt_stats, bt_stats_noTAC)


def create_backtest_summary_table(results):
    BR = []
    for model_name, _, _, _, bt_stats, bt_stats_noTAC in results:
        BR += [bt_stats_noTAC.append(pd.Series({'model': model_name, 'tac': 'N'})),
               bt_stats.append(pd.Series({'model': model_name, 'tac': 'Y'}))]
    BR = pd.concat(BR, axis = 1).transpose().set_index(['tac', 'model']).transpose()
    
    NEW_NAMES = [
            ('Mean daily return',               '01Mean return'),
            ('Mean daily standard error (NW)',  '02Standard error'),
            ('Mean daily t-statistic (NW)',     '03t-statistic'),
            ('Min daily return',                '04Minimum'),
            ('25% daily return',                '05First quartile'),
            ('Median daily return',             '06Median'),
            ('75% daily return',                '07Third quartile'),
            ('Max daily return',                '08Maximum'),
            ('share with return >= 0',          '09Share $\ge 0$'),
            ('Std daily return',                '10Standard dev.'),
            ('Skewness',                        '11Skewness'),
            ('Kurtosis',                        '12Kurtosis'),
            
            ('hist. VaR 1%',                    '201-percent VaR'),
            ('hist. VaR 5%',                    '211-percent CVaR'),
            ('hist. CVaR 1%',                   '225-percent VaR'),
            ('hist. CVaR 5%',                   '235-percent CVaR'),
            ('Max drawdown',                    '24Max. drawdown'),
            
            ('Annual return',                   '30Return'),
            ('Annual volatility',               '31Volatility'),
            ('Sharpe ratio',                    '32Sharpe ratio'),
            ('Sortino ratio',                   '33Sortino ratio'),
            ]
    
    BR.rename(index = dict(NEW_NAMES), inplace = True)
    BR = BR.loc[[o for n, o in NEW_NAMES], :].sort_index(axis = 1).astype(float)
    
    L = BR.to_latex(float_format = '%.5f', multicolumn = True)
    
    return L


def create_portfolio_value_plot(results, out_folder):
    
    fig = plt.figure(figsize = (7, 3.5))
    ax = fig.subplots(1, 1)
    
    for model_name, _, _, result, _, _ in results:
        ax.plot(result.portfolio_value / result.iloc[0].portfolio_value, 
                label = model_name[1:])
    
    #ax.set_ylabel('relative portfolio value', fontsize = 13)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    
    ax.legend(loc = 'upper left', ncol = 4, 
              fontsize = 12, frameon = False)
    
    ax.set_xlim('2013-01-01', '2019-03-31')
    
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.0, wspace = 0.0,
                        left = 0.08, right = 0.99,
                        bottom = 0.08, top = 0.99)
    fig.savefig(out_folder + '/portfolio_value.pdf')
    plt.close()



results = list(read_jobs(RUN_FOLDER, JOBS))

L = create_backtest_summary_table(results)

create_portfolio_value_plot(results, RUN_FOLDER)


