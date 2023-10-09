#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 14:53:41 2019

@author: mschnaubelt
"""

import pandas as pd
import numpy as np
from scipy.stats import logistic
from sklearn.base import BaseEstimator, ClassifierMixin


class WrappedRegressorModel(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model):
        self.base_model = base_model
    
    
    def fit(self, X, y, sample_weight = None):
        self.base_model.fit(X, y, sample_weight)
        
        self.train_y_median = y.median()
        self.train_y_std = y.std()
    
    
    def predict(self, X):
        y_pred = self.base_model.predict(X)
        
        return (y_pred > self.train_y_median)*1
    
    
    def predict_proba(self, X):
        y_pred = self.base_model.predict(X)
        
        p = logistic.cdf((y_pred - self.train_y_median) / self.train_y_std)
        p = np.stack((p, 1 - p)).transpose()
        
        return pd.DataFrame(p, index = X.index)
