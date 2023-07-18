#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:47:20 2019

@author: mschnaubelt
"""

import os
import re
import numpy as np
import pandas as pd

from util.helper import get_last_period


def load_raw_eps_data(data_dir, export_key = 'AAPL',
                      primary_ric = 'AAPL', rics = ['AAPL.OQ', 'AAPL.O']):
    
    files = ['%s/%s_eps.csv' % (data_dir, s) for s in rics]
    
    data = [pd.read_csv(f, encoding='latin') for f in files if os.path.isfile(f)]
    data = [d for d in data if 'Date' in d.columns]
    
    if len(data) == 0:
        return None
    
    data = pd.concat(data)
    
    data['Report Date'] = pd.to_datetime(data['Report Date'], utc = True, errors = 'coerce')
    data['Date'] = pd.to_datetime(data['Date'], utc = True, errors = 'coerce')
    
    data.dropna(subset = ['Report Date', 'Financial Period Absolute', 
                          'Earnings Per Share - Actual', 'Currency'], inplace = True)
    
    data.rename(index = str, 
                columns={"Earnings Per Share - Actual": "EPS Actual",
                         "Earnings Per Share - Mean": "EPS Mean Estimate",
                         "Earnings Per Share - SmartEstimate®": 
                             "EPS Smart Estimate",
                         "Earnings Per Share - Standard Deviation": 
                             "EPS Estimate Standard Deviation",
                         "Earnings Per Share - Median": 
                             "EPS Median Estimate",
                         "Earnings Per Share - Number of Estimates": 
                             "EPS Number of Estimates",
                         "Earnings Per Share - Number of Included Estimates": 
                             "EPS Number of Included Estimates",}, 
                inplace = True)
    
    data['instrument_key'] = export_key
    
    return data[['instrument_key', 'Financial Period Absolute', 'Period Month', 
                 'Period Year', 'Report Date', 'EPS Actual', 'Currency',
                 'Date', 'EPS Mean Estimate', 'EPS Median Estimate', 
                 'EPS Smart Estimate', 'EPS Estimate Standard Deviation', 
                 'EPS Number of Estimates', 'EPS Number of Included Estimates']]



def load_eps(data_dir, export_key = 'AAPL',
             primary_ric = 'AAPL', rics = ['AAPL.OQ', 'AAPL.O']):
    
    data = load_raw_eps_data(data_dir, export_key, primary_ric, rics)
    
    if data is None:
        raise Exception('No EPS data for ticker %s!' % export_key)
    
    if len(data) == 0:
        raise Exception('Empty EPS data for ticker %s, skipping!' % export_key)
    
    data.sort_values('Date', inplace = True)
    
    data.drop_duplicates(inplace = True)
    
    lasts = data.groupby(['Financial Period Absolute', 'Period Month', 
                          'Period Year']).last().reset_index()
    
    def unique_count(x):
        return len(x.unique())
    
    for col in ['EPS Mean Estimate', 'EPS Smart Estimate']:
        agg = data.groupby(['Financial Period Absolute', 'Period Month', 
                            'Period Year']).\
            aggregate({
                    'Date': ['min', 'max'],
                    col: ['min', 'max', 'first', 'std', 'count', unique_count]
                    })
        agg.columns = [' '.join(col).strip() for col in agg.columns.values]
        
        agg['Date span'] = agg['Date max'] - agg['Date min']
        
        join_cols = [col + ' ' + op for op in ['min', 'max', 'first', 'std', 
                                               'count', 'unique_count']]
        if col == 'EPS Mean Estimate':
            join_cols.append('Date span')
        
        lasts = lasts.join(agg[join_cols], on = ['Financial Period Absolute', 
                           'Period Month', 'Period Year'])
    
    
    data['last_period'] = data['Financial Period Absolute'].apply(get_last_period, mode = 'Q')
    rep_dates = data.groupby('Financial Period Absolute')['Report Date'].last()
    data['last_period_report_date'] = data.join(rep_dates, on = 'last_period', 
                            rsuffix = ' Last Period')['Report Date Last Period']
    
    estimate_available = data['Date'] <= data['last_period_report_date']
    forward_estimates = data.loc[estimate_available, ['last_period', 'EPS Mean Estimate']]
    forward_estimates.drop_duplicates(subset = ['last_period'], inplace = True, keep = 'last')
    forward_estimates = forward_estimates.set_index('last_period')
    forward_estimates.columns = ['EPS Forward Estimate']
    
    lasts = lasts.join(forward_estimates, on = 'Financial Period Absolute')
    
    
    lasts['Report Date'] = lasts['Report Date'].dt.tz_localize('UTC')
    
    return lasts[['instrument_key', 'Financial Period Absolute', 'Period Month', 
                 'Period Year', 'Report Date', 'EPS Actual', 'Currency', 
                 'Date', 'EPS Mean Estimate', 'EPS Median Estimate', 
                 'EPS Estimate Standard Deviation', 'EPS Number of Estimates', 
                 'EPS Number of Included Estimates', 'EPS Smart Estimate', 
                 'EPS Mean Estimate min', 'EPS Mean Estimate max', 
                 'EPS Mean Estimate first', 'EPS Mean Estimate std', 
                 'EPS Mean Estimate count', 'EPS Mean Estimate unique_count', 
                 'EPS Forward Estimate', 'Date span', 
                 'EPS Smart Estimate min', 'EPS Smart Estimate max', 
                 'EPS Smart Estimate first', 'EPS Smart Estimate std', 
                 'EPS Smart Estimate count', 'EPS Smart Estimate unique_count']]
    


#def load_raw_revenue_data(data_dir, export_key = 'AAPL',
#                      primary_ric = 'AAPL', rics = ['AAPL.OQ', 'AAPL.O']):
#    
#    files = ['%s/%s_revenue.csv' % (data_dir, s) for s in rics]
#    
#    data = [pd.read_csv(f, encoding='latin') for f in files if os.path.isfile(f)]
#    data = [d for d in data if 'Date' in d.columns]
#    
#    if len(data) == 0:
#        return None
#    
#    data = pd.concat(data)
#    
#    data['Report Date'] = pd.to_datetime(data['Report Date'], utc = True)
#    data['Date'] = pd.to_datetime(data['Date'], utc = True)
#    
#    data.dropna(subset = ['Report Date', 'Financial Period Absolute', 
#                          'Revenue - Actual', 'Currency'], inplace = True)
#    
#    data.rename(index = str, 
#                columns={"Revenue - Actual": "Revenue Actual",
#                         "Revenue - Mean": "Revenue Mean Estimate",
#                         "Revenue - Estimate Diffusion": "Revenue Estimate Diffusion",
#                         "Revenue - Standard Deviation": "Revenue Estimate StD",
#                         "Revenue - Number of Estimates": "Revenue NEstimates",
#                         "Revenue - Number of Included Estimates": "Revenue NIncEstimates",
#                         }, 
#                inplace = True)
#    
#    data['instrument_key'] = export_key
#    
#    return data[['instrument_key', 'Financial Period Absolute', 'Period Month', 
#                 'Period Year', 'Report Date', 'Revenue Actual', 'Currency',
#                 'Date', 'Revenue Mean Estimate', 
#                 'Revenue Estimate StD', 'Revenue NEstimates', 'Revenue NIncEstimates']]
#
#
#
#def load_revenue(data_dir, export_key = 'AAPL',
#             primary_ric = 'AAPL', rics = ['AAPL.OQ', 'AAPL.O']):
#    
#    data = load_raw_revenue_data(data_dir, export_key, primary_ric, rics)
#    
#    if data is None:
#        raise Exception('No revenue data for ticker %s!' % export_key)
#    
#    data.sort_values('Date', inplace = True)
#    
#    lasts = data.groupby(['Financial Period Absolute', 'Period Month', 
#                          'Period Year']).last().reset_index()
#    
#    return lasts[['instrument_key', 'Financial Period Absolute', 'Period Month', 
#                 'Period Year', 'Report Date', 'Revenue Actual', 'Currency',
#                 'Date', 'Revenue Mean Estimate', 
#                 'Revenue Estimate StD', 'Revenue NEstimates', 'Revenue NIncEstimates']]
    

def get_fundamental_columns(prefix, data_cols):
     cols = ['instrument_key', 'Financial Period Absolute', 'Period Month', 
            'Period Year', 'Report Date', 'Currency', prefix + ' Estimate Date', 
            prefix + ' Actual', prefix + ' Estimate Median',
            prefix + ' Estimate Mean', prefix + ' Estimate StD', 
            prefix + ' NEstimates', prefix + ' NIncEstimates',
            prefix + ' Estimate Mean std', prefix + ' Estimate Mean count',
            prefix + ' Forward Estimate']
     return [c for c in cols if c in data_cols.columns]


def load_raw_fundamental_data(data_dir, 
                              file_suffix = 'bvps', eikon_name = 'Book Value Per Share', 
                              export_key = 'AAPL',
                              primary_ric = 'AAPL', rics = ['AAPL.OQ', 'AAPL.O']):
    
    files = ['%s/%s_%s.csv' % (data_dir, s, file_suffix) for s in rics]
    
    data = [pd.read_csv(f, encoding='latin') for f in files if os.path.isfile(f)]
    data = [d for d in data if 'Date' in d.columns]
    
    if len(data) == 0:
        return None
    
    data = pd.concat(data)
    
    data['Report Date'] = pd.to_datetime(data['Report Date'], utc = True, errors = 'coerce')
    data['Date'] = pd.to_datetime(data['Date'], utc = True, errors = 'coerce')
    
    data.dropna(subset = ['Report Date', 'Financial Period Absolute', 
                          eikon_name + ' - Actual'], inplace = True)
    
    prefix = file_suffix.upper()
    
    data.rename(index = str, 
                columns = {
                        "Date": prefix + " Estimate Date",
                        eikon_name + " - Actual": prefix + " Actual",
                        eikon_name + " - Mean": prefix + " Estimate Mean",
                        eikon_name + " - Median": prefix + " Estimate Median",
                        eikon_name + " - Estimate Diffusion": prefix + " Estimate Diffusion",
                        eikon_name + " - Standard Deviation": prefix + " Estimate StD",
                        eikon_name + " - Number of Estimates": prefix + " NEstimates",
                        eikon_name + " - Number of Included Estimates": prefix + " NIncEstimates",
                        }, 
                inplace = True)
    
    data['instrument_key'] = export_key
    
    return data[get_fundamental_columns(prefix, data)]


def load_fundamental(data_dir, 
                     file_suffix = 'bvps', eikon_name = 'Book Value Per Share', 
                     export_key = 'AAPL',
                     primary_ric = 'AAPL', rics = ['AAPL.OQ', 'AAPL.O']):
    
    data = load_raw_fundamental_data(data_dir, file_suffix, eikon_name,
                                     export_key, primary_ric, rics)
    
    if data is None:
        raise Exception('No %s data for ticker %s!' % (file_suffix, export_key))
    
    if len(data) == 0:
        raise Exception('Empty %s data for ticker %s, skipping!' % (
                file_suffix, export_key))
    
    if 'Period Month' not in data.columns:
        data['Period Month'] = 0
        data['Period Year'] = 0
    
    prefix = file_suffix.upper()
    
    data.sort_values(prefix + ' Estimate Date', inplace = True)
    
    data.drop_duplicates(inplace = True)
    
    lasts = data.groupby(['Financial Period Absolute', 'Period Month', 
                          'Period Year']).last().reset_index()
    
    if (prefix + ' Estimate Mean') in data.columns:
        agg = data.groupby(['Financial Period Absolute', 'Period Month', 
                            'Period Year']).\
            aggregate({prefix + ' Estimate Mean': ['std', 'count']})
        agg.columns = [' '.join(col).strip() for col in agg.columns.values]
        
        lasts = lasts.join(agg, on = ['Financial Period Absolute', 
                                      'Period Month', 'Period Year'])
        
        
        data['last_period'] = data['Financial Period Absolute'].apply(get_last_period, mode = 'Q')
        rep_dates = data.groupby('Financial Period Absolute')['Report Date'].last()
        data['last_period_report_date'] = data.join(rep_dates, on = 'last_period', 
                                rsuffix = ' Last Period')['Report Date Last Period']
        
        estimate_available = data[prefix + ' Estimate Date'] <= data['last_period_report_date']
        forward_estimates = data.loc[estimate_available, ['last_period', prefix + ' Estimate Mean']]
        forward_estimates.drop_duplicates(subset = ['last_period'], inplace = True, keep = 'last')
        forward_estimates = forward_estimates.set_index('last_period')
        forward_estimates.columns = [prefix + ' Forward Estimate']
        
        lasts = lasts.join(forward_estimates, on = 'Financial Period Absolute')
    
    
    lasts['Report Date'] = lasts['Report Date'].dt.tz_localize('UTC')
    
    return lasts[get_fundamental_columns(prefix, lasts)]


#def load_raw_bvps_data(data_dir, export_key = 'AAPL',
#                      primary_ric = 'AAPL', rics = ['AAPL.OQ', 'AAPL.O']):
#    
#    files = ['%s/%s_bvps.csv' % (data_dir, s) for s in rics]
#    
#    data = [pd.read_csv(f, encoding='latin') for f in files if os.path.isfile(f)]
#    data = [d for d in data if 'Date' in d.columns]
#    
#    if len(data) == 0:
#        return None
#    
#    data = pd.concat(data)
#    
#    data['Report Date'] = pd.to_datetime(data['Report Date'], utc = True)
#    data['Date'] = pd.to_datetime(data['Date'], utc = True)
#    
#    data.dropna(subset = ['Report Date', 'Financial Period Absolute', 
#                          'Book Value Per Share - Actual', 'Currency'], inplace = True)
#    
#    data.rename(index = str, 
#                columns={"Book Value Per Share - Actual": "BVPS Actual",
#                         "Book Value Per Share - Mean": "BVPS Mean Estimate",
#                         "Book Value Per Share - Estimate Diffusion": "BVPS Estimate Diffusion",
#                         "Book Value Per Share - Standard Deviation": "BVPS Estimate StD",
#                         "Book Value Per Share - Number of Estimates": "BVPS NEstimates",
#                         "Book Value Per Share - Number of Included Estimates": "BVPS NIncEstimates",
#                         }, 
#                inplace = True)
#    
#    data['instrument_key'] = export_key
#    
#    return data[['instrument_key', 'Financial Period Absolute', 'Period Month', 
#                 'Period Year', 'Report Date', 'BVPS Actual', 'Currency',
#                 'Date', 'BVPS Mean Estimate', 
#                 'BVPS Estimate StD', 'BVPS NEstimates', 'BVPS NIncEstimates']]
#
#
#def load_bvps(data_dir, export_key = 'AAPL',
#             primary_ric = 'AAPL', rics = ['AAPL.OQ', 'AAPL.O']):
#    
#    data = load_raw_bvps_data(data_dir, export_key, primary_ric, rics)
#    
#    if data is None:
#        raise Exception('No BVPS data for ticker %s!' % export_key)
#    
#    data.sort_values('Date', inplace = True)
#    
#    lasts = data.groupby(['Financial Period Absolute', 'Period Month', 
#                          'Period Year']).last().reset_index()
#    
#    return lasts[['instrument_key', 'Financial Period Absolute', 'Period Month', 
#                 'Period Year', 'Report Date', 'BVPS Actual', 'Currency',
#                 'Date', 'BVPS Mean Estimate', 
#                 'BVPS Estimate StD', 'BVPS NEstimates', 'BVPS NIncEstimates']]



def load_raw_event_data(data_dir, export_key = 'AAPL',
                        primary_ric = 'AAPL', rics = ['AAPL.OQ', 'AAPL.O']):
    
    files = ['%s/%s_event.csv' % (data_dir, s) for s in rics]
    
    data = [pd.read_csv(f, encoding='latin') for f in files if os.path.isfile(f)]
    data = [d for d in data if 'Date' in d.columns]
    
    if len(data) == 0:
        return None
    
    data = pd.concat(data)
    
    data['Date'] = pd.to_datetime(data['Date'], utc = True)
    
    data.dropna(subset = ['Date', 'Company Event Type'], inplace = True)
    data.drop_duplicates(subset = ['Date', 'Company Event Type', 'Event Title'],
                         inplace = True)
    data.sort_values('Date', inplace = True)
    
    data['instrument_key'] = export_key
    
    return data[['instrument_key', 'Date', 'Company Event Type', 'Event Title']]



def load_events(data_dir, export_key = 'AAPL',
                primary_ric = 'AAPL', rics = ['AAPL.OQ', 'AAPL.O']):
    
    data = load_raw_event_data(data_dir, export_key, primary_ric, rics)
    
    if data is None or len(data) == 0:
        raise Exception('No event data for ticker %s!' % export_key)
    
    pat_1 = re.compile('Q([1234]) ([0-9]{2}|[0-9]{4}) ', re.IGNORECASE)
    
    def matcher_1(x):
        m = re.search(pat_1, x)
        
        if m is None or len(m.groups()) != 2:
            return pd.Series(index = ['f_quarter', 'f_year'])
        
        quarter = int(m.groups()[0])
        year = int(m.groups()[1])
        
        if year < 1000:
            year += 2000
        
        return pd.Series({'f_quarter': quarter, 'f_year': year})
    
    
    pat_2 = re.compile('FY[\W]?([0-9]{2}|[0-9]{4}) (First|Second|Third|Fourth) Quarter', 
                       re.IGNORECASE)
    
    def matcher_2(x):
        m = re.search(pat_2, x)
        
        if m is None or len(m.groups()) != 2:
            return pd.Series(index = ['f_quarter', 'f_year'])
        
        quarter = m.groups()[1]
        year = int(m.groups()[0])
        
        quarter = {'First': 1, 'Second': 2, 'Third': 3, 'Fourth': 4}[quarter]
        
        if year < 1000:
            year += 2000
        
        return pd.Series({'f_quarter': quarter, 'f_year': year})
    
    qy_matches_1 = data['Event Title'].apply(matcher_1)
    qy_matches_2 = data['Event Title'].apply(matcher_2)
    
    data['f_year'] = np.where(pd.isna(qy_matches_1.f_year), 
        qy_matches_2.f_year, qy_matches_1.f_year)
    
    data['f_quarter'] = np.where(pd.isna(qy_matches_1.f_quarter), 
        qy_matches_2.f_quarter, qy_matches_1.f_quarter)
    
    
    def fiscal_period(r):
        if not (pd.isna(r['f_year']) or pd.isna(r['f_quarter'])):
            return 'FY%4dQ%d' % (r['f_year'], r['f_quarter'])
        else:
            return None
    
    data['fiscal_period'] = data.apply(fiscal_period, axis = 1)
    
    return data
    


def load_raw_marketcap_data(data_dir, export_key = 'AAPL',
                            primary_ric = 'AAPL.O', rics = ['AAPL.OQ', 'AAPL.O']):
    
    files = ['%s/%s_marketcap.csv' % (data_dir, s) for s in rics]
    
    data = [pd.read_csv(f) for f in files if os.path.isfile(f)]
    data = [d for d in data if 'Price Close' in d.columns]
    
    if len(data) == 0:
        return None
    
    data = pd.concat(data)
    data.dropna(subset = ['Date', 'Price Close'], inplace = True)
    
    data['Date'] = pd.to_datetime(data['Date'], utc = True)
    
    data['is_primary'] = (data.Instrument == primary_ric)*1.0
    data['instrument_key'] = export_key
    
    data.sort_values('is_primary', ascending = False, inplace = True)
    
    data.drop_duplicates(subset = ['Date'],
                         inplace = True)
    
    data.sort_values('Date', ascending = True, inplace = True)
    
    return data



def load_marketcap(data_dir, export_key, primary_ric, rics):
    mkt_data = load_raw_marketcap_data(data_dir, export_key, primary_ric, rics)
    
    if mkt_data is None or len(mkt_data) == 0:
        raise Exception('No market cap data for ticker %s!' % export_key)
    
    mkt_data['Outstanding Shares Adj'] = mkt_data['Company Market Cap'] / mkt_data['Price Close']
    mkt_data['Price To Book Value Per Share'] = \
        mkt_data['Price To Book Value Per Share (Daily Time Series Ratio)']
    
    mkt_data['instrument_key'] = export_key
    
    mkt_data = mkt_data[['instrument_key', 'Date', 'Currency', 'Company Market Cap',
                         'Outstanding Shares', 'Outstanding Shares Adj',
                         'Price To Book Value Per Share', 'Volume']]
    
    return mkt_data.reset_index(drop = True)



def load_raw_fundratio_data(data_dir, export_key = 'AAPL',
                            primary_ric = 'AAPL.O', rics = ['AAPL.OQ', 'AAPL.O']):
    
    files = ['%s/%s_fundratio.csv' % (data_dir, s) for s in rics]
    
    data = [pd.read_csv(f) for f in files if os.path.isfile(f)]
    data = [d for d in data if 'Currency' in d.columns]
    
    if len(data) == 0:
        return None
    
    data = pd.concat(data)
    data.dropna(subset = ['Date', 'Currency'], inplace = True)
    
    data['Date'] = pd.to_datetime(data['Date'], utc = True)
    
    data['is_primary'] = (data.Instrument == primary_ric)*1.0
    data['instrument_key'] = export_key
    
    data.sort_values('is_primary', ascending = False, inplace = True)
    
    data.drop_duplicates(subset = ['Date'],
                         inplace = True)
    
    data.sort_values('Date', ascending = True, inplace = True)
    
    return data


def load_fundratio(data_dir, export_key, primary_ric, rics):
    fund_data = load_raw_fundratio_data(data_dir, export_key, primary_ric, rics)
    
    if fund_data is None or len(fund_data) == 0:
        raise Exception('No fundamental ratio data for ticker %s!' % export_key)
    
    fund_data['instrument_key'] = export_key
    
    fund_data.columns = [c.replace(' (Daily Time Series Ratio)', '') for c in fund_data.columns]
    fund_data.columns = [c.replace(' (Daily Time Series)', '') for c in fund_data.columns]
    
    fund_data = fund_data[['instrument_key', 'Date', 'Currency', 'Enterprise Value',
                           'Company Market Cap', 'Outstanding Shares',
                           'Price To Book Value Per Share',
                           'Price To Tangible Book Value Per Share', 'P/E', 'P/E/G',
                           'Enterprise Value To EBIT', 'Enterprise Value To EBITDA',
                           'Price To Sales Per Share', 'Enterprise Value To Sales',
                           'Price To Cash Flow Per Share',
                           'Enterprise Value To Operating Cash Flow',
                           'Total Debt To Enterprise Value', 'Net Debt To Enterprise Value',
                           'Dividend yield', 'Net Dividend Yield']]
    
    return fund_data.reset_index(drop = True)



