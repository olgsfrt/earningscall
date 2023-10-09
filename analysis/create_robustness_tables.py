#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:36:10 2020

@author: mschnaubelt
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import FINAL_SEED_RUN, FINAL_PARAMETER_RUN
from analysis_helper import extract_job_info, read_run

from return_distribution import analyze_return_by_tr_sector, extended_describe
from create_tables import extract_model_comparison_column


def process_seed_run(run_folder):
    
    results = read_run(run_folder)
    results = results.apply(extract_model_comparison_column, axis = 1)
    
    SEL_COLS = ['02Mean', '03t-statistic', '06Median', '12Dir. Acc.', 
                '51Count top-flop', '52Mean top-flop', '53t-statistic', 
                '56Median top-flop', '62Dir. acc. top-flop']
    
    SR = results[SEL_COLS].groupby(level = 1).agg(['mean', 'min', 'max', 'std', lambda df: df.iloc[0]])
    SR = SR.stack(level = 0)[['min', 'max', 'mean', 'std', '<lambda>']]
    SR = SR.unstack(0).swaplevel(0, 1, axis = 1).sort_index(axis = 1)
    SR.columns = SR.columns.set_levels(['5baseline', '2max', '3mean', '1min', '4std'], level = 1)
    SR = SR.sort_index(axis = 1)
    
    
    SR.to_excel(run_folder + '/seed_table.xlsx')
    
    L = SR.to_latex(float_format = '%.4f', multicolumn = True)
    L = re.sub('[ ]+', ' ', L)
    with open(run_folder + '/seed_table.tex', "w") as f:
        print(L, file = f)


def process_parameter_run(run_folder):
    
    raw_results = read_run(run_folder)
    win_size = raw_results['job'].apply(lambda j: j['rolling_window_size'])
    raw_results['window_size'] = win_size
    results = raw_results.apply(extract_model_comparison_column, axis = 1)
    results['window_size'] = raw_results['window_size']
    results.set_index('window_size', append = True, inplace = True)
    
    SEL_COLS = ['02Mean', '12Dir. Acc.', 
                '51Count top-flop', '52Mean top-flop', 
                '62Dir. acc. top-flop']
    SR = results[SEL_COLS]
    
    SR = SR.unstack('period')
    SR = SR.swaplevel(0, 1, axis = 1).sort_index(axis = 1)
    SR.reset_index(level = 1, drop = True, inplace = True)
    
    def fmter(x):
        if pd.isna(x):
            return '--'
        else:
            return '%.3f' % x if abs(x) < 1000 else '%d' % x
    
    SR.to_excel(run_folder + '/parameter_table.xlsx')
    
    L = SR.to_latex(float_format = fmter, multicolumn = True)
    L = re.sub('[ ]+', ' ', L)
    with open(run_folder + '/parameter_table.tex', "w") as f:
        print(L, file = f)


if __name__ == '__main__':
    process_seed_run(run_folder = FINAL_SEED_RUN)
    process_parameter_run(run_folder = FINAL_PARAMETER_RUN)

    