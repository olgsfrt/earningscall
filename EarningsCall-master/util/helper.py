#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:31:07 2019

@author: mschnaubelt
"""

import re
import numpy as np

def get_last_period(p, mode = 'Y'):
    m = re.search(re.compile('FY([0-9]{2}|[0-9]{4})Q([1234])', re.IGNORECASE), p)
    
    if m is None or len(m.groups()) != 2:
        return np.nan
    
    if mode == 'Y':
        year = int(m.groups()[0]) - 1
        quarter = int(m.groups()[1])
    elif mode == 'Q':
        year = int(m.groups()[0])
        quarter = int(m.groups()[1]) - 1
        if quarter == 0:
            year -= 1
            quarter = 4
    else:
        return np.nan
    
    return 'FY%dQ%d' % (year, quarter)

def get_next_quarter(p):
    m = re.search(re.compile('FY([0-9]{2}|[0-9]{4})Q([1234])', re.IGNORECASE), p)
    
    if m is None or len(m.groups()) != 2:
        return np.nan
    
    year = int(m.groups()[0])
    quarter = int(m.groups()[1]) + 1
    if quarter == 5:
        year += 1
        quarter = 1
    
    return 'FY%dQ%d' % (year, quarter)
