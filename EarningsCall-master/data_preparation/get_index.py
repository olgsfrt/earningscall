# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:23:20 2019

@author: eikon
"""

import time
import numpy as np
import pandas as pd
import eikon as ek
import json


out_dir = 'C:/Users/eikon.WSBI-38/Desktop/EC/'

#INDEX = '0#.SPX'
#INDEX = '0#.SP400'
INDEX = '0#.SPCY'
#INDEX = '.RUI'

ek.set_app_key('ac2e022cefad4af595f4517611492f7c9d1b918a')



dates = list(pd.date_range('1999-01-01', '2019-06-01', freq = '3M').strftime('%Y-%m-%d'))

ric_fields = [ek.TR_Field('TR.RIC')]
id_fields = [ek.TR_Field('TR.InstrumentID'), ek.TR_Field('TR.OrganizationID')]

result = []

for date in dates:
    print("Requesting index %s for date %s" % (INDEX, date))
    ric_data, err = ek.get_data([INDEX], ric_fields, {'SDate': date})
    
    id_data, err = ek.get_data([INDEX], id_fields, {'SDate': date})
    
    data = {}
    
    data['Date'] = date
    data['Index'] = INDEX
    data['Instruments'] = list(ric_data.Instrument.values)
    data['RICs'] = list(ric_data.RIC.values)
    data['InstrumentIDs'] = list(id_data['Instrument ID'].astype(str).values)
    data['OrganizationIDs'] = list(id_data['Organization PermID'].astype(str).values)
    
    result.append(data)
    
    time.sleep(0.5)


with open('%s/index_%s.json' % (out_dir, INDEX), "w") as data_file:
    json.dump(result, data_file, indent = 4, sort_keys = True)

