#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:47:26 2020

@author: mschnaubelt
"""

SCRIPTS = ['create_tables.py', 'plot_decile_return_matrix.py', 'mean_difference_significance.py']

for script in SCRIPTS:
    with open(script, 'r') as f:
        exec(f.read())
