#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:57:16 2019

@author: mschnaubelt
"""

import pandas as pd
import numpy as np


INTERNATIONAL_EXCHANGES = ['.TO', '.V', '.L', '.BN', '.S', '.F', '.MU', '.HK', '.DE', 
                           '.V', '.PA', '.MA', '.CO', '.HE', '.AX', '.MI']



def create_data_index(symbol_list_file):
    data = pd.read_excel(symbol_list_file)
    data.dropna(axis = 0, subset = ['sa_ticker'], inplace = True)
    
    data.loc[data['TR.InstrumentID'].str.contains('Unable').fillna(True), 
             'TR.InstrumentID'] = np.nan
    data['TR.InstrumentID'] = data['TR.InstrumentID'].apply(
            lambda x: np.nan if pd.isnull(x) else str(x))
    
    def reduce_group(g):
        rics = [x for x in set(list(g['TR.RIC'].values) + list(g['TR.PrimaryRIC'].values)
                + list(g.best_ric.values)) if not pd.isnull(x)]
        
        iids = [x for x in g['TR.InstrumentID'].unique() if not pd.isnull(x)]
        
        primary_ric = g['TR.PrimaryRIC'].unique()[0]
        countries = g['TR.RegCountryCode'].unique()
        type_codes = g['TR.InstrumentTypeCode'].unique()
        share_classes = g['TR.ShareClass'].unique()	
        
        company_name = g['TR.CompanyName'].unique()
        
        naics = g['TR.NAICSNationalIndustryCode'].unique()
        naics_sec = g['TR.NAICSSector'].unique()
        naics_subsec = g['TR.NAICSSubsector'].unique()
        
        trbc = g['TR.TRBCIndustryCode'].unique()
        trbc_sec = g['TR.TRBCEconomicSector'].unique()
        
        type_codes = [tc for tc in type_codes if not pd.isna(tc) and
                          not any([fc in tc for fc in ['Unable to collect','nan']])]
        
        share_classes = [sc for sc in share_classes if not pd.isna(sc) and
                          not any([fc in sc for fc in ['Unable to collect','nan']])]
        
        return pd.Series({
             'sa_ticker': g.name,
             'rics': rics,
             'instrument_ids': iids,
             'instrument_id': iids[0] if len(iids) == 1 else np.nan,
             'primary_ric': rics[0] if pd.isnull(primary_ric) else primary_ric,
             'ids:len': len(iids),
             'rics:len': len(rics),
             'country': countries[0] if len(countries) == 1 else np.nan,
             'type_code': type_codes[0] if len(type_codes) == 1 else np.nan,
             'share_class': share_classes[0] if len(share_classes) == 1 else np.nan,
             
             'company_name': company_name[0] if len(company_name) == 1 else np.nan,
             
             'naics': naics[0] if len(naics) == 1 else np.nan,
             'naics_sec': naics_sec[0] if len(naics_sec) == 1 else np.nan,
             'naics_subsec': naics_subsec[0] if len(naics_subsec) == 1 else np.nan,
             'trbc': trbc[0] if len(trbc) == 1 else np.nan,
             'trbc_sec': trbc_sec[0] if len(trbc_sec) == 1 else np.nan,
             })
    
    red = data.groupby(['sa_ticker']).apply(reduce_group)
    
    return red

