#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:26:18 2019

@author: mschnaubelt
"""

import pandas as pd
import numpy as np
from scipy.stats import logistic
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class EarningsSurpriseModel(BaseEstimator, ClassifierMixin):
    def __init__(self, surprise_column):
        self.surprise_column = surprise_column
    
    
    def fit(self, X, y, sample_weight=None):
        self.std = X[self.surprise_column].std()
        self.median = X[self.surprise_column].median()

    
    
    def predict(self, X):
        
        return (X[self.surprise_column] > self.median)*1
    
    
    def predict_proba(self, X):
        
        p = logistic.cdf((X[self.surprise_column] - self.median) / self.std)
        p = np.stack((1 - p, p)).transpose()
        
        return pd.DataFrame(p, index = X.index)


class EarningsSurpriseRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, surprise_column):
        self.surprise_column = surprise_column
    
    
    def fit(self, X, y, sample_weight=None):
        
        self.median = X[self.surprise_column].median()
        self.std_X = X[self.surprise_column].std()
        self.std_y = y.std()
    
    def predict(self, X):
        r = X[self.surprise_column]
        r.name = None
        return (r - self.median) / self.std_X * self.std_y
    
