#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:34:26 2019

@author: mschnaubelt
"""

import pandas as pd
import statsmodels.api as sm
import pytz

import os

os.getcwd()

os.chdir('c:\\Users\\aq75iwit\\Anaconda3\\envs\\earnings_call_7\\EarningsCall')


def analyze_return_by_year(predictions, output_basefilename = None):
    
    years = predictions.local_date.str[0:4].astype(int)
    
    grouped_returns = (predictions.signed_target_return*1E4).groupby(
            [predictions.is_top_flop, years]).describe()
    
    all_ret = (predictions.signed_target_return*1E4).groupby(years).describe()
    
    result = pd.concat([all_ret, grouped_returns])
    
    if output_basefilename:
        result.to_csv(output_basefilename + '.csv')
    
    return result


def analyze_return_by_period(predictions, output_basefilename = None):
    
    grouped_returns = (predictions.signed_target_return*1E4).groupby(
            [predictions.is_top_flop, predictions.fiscal_period]).describe()
    
    all_ret = (predictions.signed_target_return*1E4).groupby(predictions.fiscal_period).describe()
    
    result = pd.concat([all_ret, grouped_returns])
    
    if output_basefilename:
        result.to_csv(output_basefilename + '.csv')
    
    return result


def analyze_return_by_hour(predictions, output_basefilename = None):

    #predictions.release_datetime = pd.to_datetime(predictions.release_datetime)

    
    #hours = predictions.release_datetime.dt.tz_convert('America/New_York').dt.hour

    predictions.release_datetime = pd.to_datetime(predictions.release_datetime)
    predictions.release_datetime = predictions.release_datetime.dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    hours = predictions.release_datetime.dt.hour

    
    grouped_returns = (predictions.signed_target_return*1E4).groupby(
            [predictions.is_top_flop, hours]).describe()
    
    all_ret = (predictions.signed_target_return*1E4).groupby(hours).describe()
    
    result = pd.concat([all_ret, grouped_returns])
    
    if output_basefilename:
        result.to_csv(output_basefilename + '.csv')
    
    return result


def analyze_return_by_report_diff(predictions, output_basefilename = None):

    #predictions.release_datetime = pd.to_datetime(predictions.release_datetime)

    
    #dates = pd.to_datetime(predictions['EPS Report Date'].dt.tz_convert('America/New_York').dt.date)

    predictions['EPS Report Date'] = pd.to_datetime(predictions['EPS Report Date'])
    predictions['EPS Report Date'] = predictions['EPS Report Date'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    dates = pd.to_datetime(predictions['EPS Report Date'].dt.date)

    
    diff = dates - predictions['final_datetime']
    
    grouped_returns = (predictions.signed_target_return*1E4).groupby(
            [predictions.is_top_flop, diff]).describe()
    
    all_ret = (predictions.signed_target_return*1E4).groupby(diff).describe()
    
    result = pd.concat([all_ret, grouped_returns])
    
    if output_basefilename:
        result.to_csv(output_basefilename + '.csv')
    
    return result



def extended_describe(ser):
    r = ser.describe()
    
    X = pd.Series(1, index = ser.index)
    
    reg = sm.OLS(X, ser).fit(cov_type = 'HAC', cov_kwds = {'maxlags': 10})
    t = reg.params[0] / reg.HC0_se[0]
    
    r['t'] = t
    r['std_err'] = reg.HC0_se[0]
    
    return r


def analyze_return_by_tr_sector(predictions, output_basefilename = None):
    
    grouped_returns = (predictions.signed_target_return*1E2).groupby(
            [predictions.is_top_flop, predictions.trbc_sec]).apply(extended_describe)
    #grouped_returns.index.set_levels(['no top-flop', 'top-flop'], level = 0, inplace = True)
    grouped_returns.index = grouped_returns.index.set_levels(['no top-flop', 'top-flop'], level = 0)

    
    all_ret = (predictions.signed_target_return*1E2).groupby(predictions.trbc_sec).apply(extended_describe)
    all_ret = pd.concat([all_ret], keys = ['all'], names = ['is_top_flop'])
    
    result = pd.concat([all_ret, grouped_returns]).unstack(level = 2)
    
    if output_basefilename:
        result.to_csv(output_basefilename + '.csv')
    
    return result


def analyze_return_by_naics_sector(predictions, output_basefilename = None):
    
    grouped_returns = (predictions.signed_target_return*1E4).groupby(
            [predictions.is_top_flop, predictions.naics_sec]).describe()
    
    all_ret = (predictions.signed_target_return*1E4).groupby(predictions.naics_sec).describe()
    
    result = pd.concat([all_ret, grouped_returns])
    
    if output_basefilename:
        result.to_csv(output_basefilename + '.csv')
    
    return result


def analyze_return_by_subsector(predictions, output_basefilename = None):
    
    result = (predictions.signed_target_return*1E4).groupby([predictions.naics_sec, predictions.naics_subsec]).describe()
    
    if output_basefilename:
        result.to_csv(output_basefilename + '.csv')
    
    return result


def analyze_return_by_value(predictions, output_basefilename = None):
    
    decile = pd.qcut(predictions.MV_log, 10)
    
    grouped_returns = (predictions.signed_target_return*1E4).groupby(
            [predictions.is_top_flop, decile]).describe()
    
    all_ret = (predictions.signed_target_return*1E4).groupby(decile).describe()
    
    result = pd.concat([all_ret.reset_index(), grouped_returns.reset_index()], 
                       ignore_index = True)
    
    result.set_index(['is_top_flop', 'MV_log'], inplace = True)
    
    if output_basefilename:
        result.to_csv(output_basefilename + '.csv')
    
    return result


def analyze_return_by_bm_ratio(predictions, output_basefilename = None):
    
    decile = pd.qcut(predictions.BM_ratio, 10)
    
    grouped_returns = (predictions.signed_target_return*1E4).groupby(
            [predictions.is_top_flop, decile]).describe()
    
    all_ret = (predictions.signed_target_return*1E4).groupby(decile).describe()
    
    result = pd.concat([all_ret.reset_index(), grouped_returns.reset_index()], 
                       ignore_index = True)
    
    result.set_index(['is_top_flop', 'BM_ratio'], inplace = True)
    
    if output_basefilename:
        result.to_csv(output_basefilename + '.csv')
    
    return result


def analyze_return(predictions, alpha = 0.1, output_basefilename = None):
    predictions['is_top_flop'] = (predictions.pred_rank < alpha) | (predictions.pred_rank > 1 - alpha)
    
    by_year = analyze_return_by_year(predictions, output_basefilename + '_by_year')
    by_period = analyze_return_by_period(predictions, output_basefilename + '_by_period')
    by_hour = analyze_return_by_hour(predictions, output_basefilename + '_by_hour')
    by_report_diff = analyze_return_by_report_diff(predictions, output_basefilename + '_by_report_diff')
    by_tr_sector = analyze_return_by_tr_sector(predictions, output_basefilename + '_by_tr_sector')
    by_naics_sector = analyze_return_by_naics_sector(predictions, output_basefilename + '_by_naics_sector')
    by_subsector = analyze_return_by_subsector(predictions, output_basefilename + '_by_subsector')
    
    try:
        by_value = analyze_return_by_value(predictions, output_basefilename + '_by_value')
    except:
        by_value = None
    
    try:
        by_bm_ratio = analyze_return_by_bm_ratio(predictions, output_basefilename + '_by_bm_ratio')
    except:
        by_bm_ratio = None
    
    return by_year, by_period, by_hour, by_report_diff, by_tr_sector, by_naics_sector, by_subsector, by_value, by_bm_ratio


if __name__ == '__main__':
    pass



