#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:05:42 2019

@author: mschnaubelt
"""

import numpy as np
from sklearn.utils import check_consistent_length


def return_score(y_true, y_pred, abnormal_return = None):
    """Return classification score.
    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.
    Read more in the :ref:`User Guide <accuracy_score>`.
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    abnormal_return : array-like of shape = [n_samples]
        Sample weights, which should be abnormal returns.
    Returns
    -------
    score : float
        Mean abnormal return.
    """
    
    check_consistent_length(y_true, y_pred, abnormal_return)
    
    signed_return = (2.0 * y_pred - 1) * abnormal_return
    
    return np.mean(signed_return)

