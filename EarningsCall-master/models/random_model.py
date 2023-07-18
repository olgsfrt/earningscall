#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 14:53:19 2019

@author: mschnaubelt
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class MonkeyModel(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.Ps = pd.DataFrame()
    
    
    def fit(self, X, y, sample_weight=None):
        pass
    
    
    def _get_probas(self, X):
        try:
            return self.Ps.loc[X.index]
        except:
            p = np.random.normal(0.5, 0.1, size = len(X))
            p = np.stack((1 - p, p)).transpose()
            
            self.Ps = self.Ps.append(pd.DataFrame(p, index = X.index))
            
            return self.Ps.loc[X.index]
    
    
    def predict(self, X):
        
        return self._get_probas(X).idxmax(axis = 1)
    
    
    def predict_proba(self, X):
        
        return self._get_probas(X)

