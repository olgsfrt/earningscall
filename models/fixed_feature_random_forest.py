#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:35:40 2019

@author: mschnaubelt
"""

from sklearn.tree import DecisionTreeClassifier
import numpy as np
class FixedFeatureRFC:
    def __init__(self, n_estimators=1000, random_state=0):
        self.n_estimators = n_estimators

        if random_state is None:
            self.random_state = np.random.RandomState()

    def fit(self, X, y, feats_fixed=None, max_features=None, bootstrap_frac=0.8):
        """
        feats_fixed: indices of features (columns of X) to be 
                     always used to train each estimator

        max_features: number of features that each estimator will use,
                      including the fixed features.

        bootstrap_frac: size of bootstrap sample that each estimator will use.
        """
        self.estimators = []
        self.feats_used = []
        self.n_classes  = np.unique(y).shape[0]

        if feats_fixed is None:
            feats_fixed = []
        if max_features is None:
            max_features = X.shape[1]

        n_samples = X.shape[0]
        n_bs = int(bootstrap_frac*n_samples)

        feats_fixed = list(feats_fixed)
        feats_all   = range(X.shape[1])

        random_choice_size = max_features - len(feats_fixed)

        feats_choosable = set(feats_all).difference(set(feats_fixed))
        feats_choosable = np.array(list(feats_choosable))

        for i in range(self.n_estimators):
            chosen = self.random_state.choice(feats_choosable,
                                              size=random_choice_size,
                                              replace=False)
            feats = feats_fixed + list(chosen)
            self.feats_used.append(feats)

            bs_sample = self.random_state.choice(n_samples,
                                                 size=n_bs,
                                                 replace=True)

            dtc = DecisionTreeClassifier(random_state=self.random_state,
                                         max_depth = 10, min_samples_split = 200,
                                       class_weight = "balanced_subsample",)
            dtc.fit(X[bs_sample][:,feats], y[bs_sample])
            self.estimators.append(dtc)

    def predict_proba(self, X):
        out = np.zeros((X.shape[0], self.n_classes))
        for i in range(self.n_estimators):
            out += self.estimators[i].predict_proba(X[:,self.feats_used[i]])
        return out / self.n_estimators

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y):
        return (self.predict(X) == y).mean() 
