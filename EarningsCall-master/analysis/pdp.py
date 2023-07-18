#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:53:30 2019

@author: mschnaubelt
"""

import pandas as pd
import numpy as np

import multiprocessing as mp

from sklearn.utils import check_array
from sklearn.base import is_classifier, is_regressor
from sklearn.inspection.partial_dependence import _grid_from_X
from sklearn.exceptions import NotFittedError



def _partial_dependence_worker(est, new_values, X, y_return, features):
    X_eval = X.copy()
    for i, variable in enumerate(features):
        X_eval.loc[:, variable] = new_values[i]
    
    try:
        if is_classifier(est):
            predictions = est.predict_proba(X_eval)[:,1]
            #returns = (2*est.predict(X_eval) - 1) * y_return
            abs_pred = np.abs(est.predict_proba(X_eval)[:,1] - 0.5)
        if is_regressor(est):
            predictions = est.predict(X_eval)
            abs_pred = np.abs(est.predict(X_eval))
    except NotFittedError:
        raise ValueError(
            "'estimator' parameter must be a fitted estimator")
    
    # average over samples
    return {
            'val': new_values[0] if len(new_values)==1 else new_values,
            'N': len(X),
            'mean_pred': np.mean(predictions, axis=0),
            'mean_abs_pred': np.mean(abs_pred, axis=0),
            #'mean_return': np.mean(returns, axis=0)
            }


def partial_dependence(est, X, y_return, features, quantiles, pool = mp.Pool(mp.cpu_count())):
    
    jobs = [(est, q, X, y_return, features) for q in quantiles]
    
    averaged_predictions = pool.starmap(_partial_dependence_worker, jobs)
    
    return pd.DataFrame(averaged_predictions)


def conditional_1d_dependence(est, X, y_return, feature, quantiles):
    
    n_features = est.n_features_
    
    features = np.asarray(features, dtype=np.int32, order='C').ravel()
    if any(not (0 <= f < n_features) for f in features):
        raise ValueError('all features must be in [0, %d]'
                         % (n_features - 1))
    
    grid = X.iloc[:, features].quantile(np.linspace(*percentiles, num = grid_resolution)).values
    grid = grid.flatten()
    
    averaged_predictions = []
    
    
    for new_values_i in range(len(grid)):
        
        lower = grid[new_values_i - 1] if new_values_i > 0 else -100
        upper = grid[new_values_i]
        
        f = (X.iloc[:, features[0]] < upper) & (X.iloc[:, features[0]] >= lower)
        X_eval = X[f]
        y_return_eval = y_return[f]
        
        try:
            if is_classifier(est):
                predictions = est.predict_proba(X_eval)[:,1]
                returns = y_return_eval
            if is_regressor(est):
                pass
        except NotFittedError:
            raise ValueError(
                "'estimator' parameter must be a fitted estimator")
        
        # average over samples
        averaged_predictions.append({
                'val_lower': lower,
                'val_upper': upper,
                'N': len(X_eval),
                'mean_pred': np.mean(predictions, axis=0),
                'mean_return': np.mean(returns, axis=0)
                })
    
    
    return pd.DataFrame(averaged_predictions)


