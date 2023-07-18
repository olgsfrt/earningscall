#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:09:16 2020

@author: mschnaubelt
"""

import re
import os
import pandas as pd

from config import FEATURE_SETS_DICT


def get_model_name(name):
    if type(name) != str:
        clas = name
        name = str(name)
    
    result = None
    
    if ('RandomForestClassifier' in name) or ('RandomForestRegressor' in name):
        result = '8 RF'
        m = re.search("max_depth[=\ ]*([0-9]+)", name)
        if m:
            d = int(m.groups()[0])
            result += '-D%d' % d
        m = re.search("n_estimators[=\ ]*([0-9]+)", name)
        if m:
            e = int(m.groups()[0])
            result += '-E%d' % e
    elif ('LogisticRegression' in name) or ('LinearRegression' in name):
        result = '1 LR'
    elif ('Lasso' in name):
        result = '3 LASSO'
    elif ('ElasticNet' in name):
        result = '5 ELASTNET'
    elif 'EarningsSurprise' in name:
        sc = clas.surprise_column
        result = '0 ES-' + sc.replace('_', '-')
    
    if 'QuantileTransformer' in name:
        result += '-Q'
    if 'KBinsDiscretizer' in name:
        m = re.search("n_bins[=\ ]*([0-9]+)", name)
        if m:
            b = int(m.groups()[0])
            result += '-B%d' % b
        else:
            result += '-B'
    if 'FunctionTransformer' in name:
        m = re.search("FunctionTransformer[\S\s]+func=\<function\ ([a-z_]*)\ at", name)
        if m:
            f = m.groups()[0]
            result += '-F%s' % f
        else:
            result += '-F'
    
    return result



def extract_job_info(R):
    C = {}
    
    C['model'] = get_model_name(R['job']['model'])
    
    m = re.search('_([-0-9]+)d_', R['job']['return_target'])
    C['period'] = int(m.groups()[0])
    
    f = set(R['job']['features'])
    f_names = [n for n, fs in FEATURE_SETS_DICT.items() if f.issuperset(fs)]
    f_names.sort()
    C['features'] = '+'.join(f_names)
    if 'POL' in C['features']:
        C['features'] += '+%d' % len(f)
    
    if 'ES' in C['model']:
        C['features'] = 'FE+POL+UIQ+VR+54'
    
    return pd.Series(C)


def read_run(run_folder):
    jobs = [f for f in os.listdir(run_folder) if f.startswith('job')]
    jobs.sort(key = lambda s: int(s.replace('job-', '')))
    
    results = [pd.read_hdf(run_folder + '%s/model_results.hdf' % f) for f in jobs 
                   if os.path.isfile(run_folder + '%s/model_results.hdf' % f)]
    results = pd.concat(results, axis = 1).transpose()
    
    results = results.join(results.apply(extract_job_info, axis = 1))
    results = results.set_index(['model', 'period', 'features'])
    
    return results


