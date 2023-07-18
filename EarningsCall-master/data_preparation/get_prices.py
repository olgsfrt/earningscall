# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:07:15 2019

@author: eikon
"""

import os
import time
import numpy as np
import pandas as pd
import eikon as ek


symbol_list_file = 'C:/Users/eikon.WSBI-38/Desktop/EC/symbol_list_excel.xlsx'

out_dir = 'C:/Users/eikon.WSBI-38/Desktop/EC/data/'

ek.set_app_key('ac2e022cefad4af595f4517611492f7c9d1b918a')

DATA_ITEMS = ['REVENUE','BVPS','EPS','MKT_CAP', 'CFPS', 'ROEA', 'DIVEST'] # ['PRICE','DIVIDEND','EPS','EVENT']


if '.hdf' in symbol_list_file:
    ticker_list = pd.read_hdf(symbol_list_file)
    ticker_list = ticker_list[ticker_list.error.isnull()]
    ticker_list.sort_values('best_ric', inplace = True)
else:
    ticker_list = pd.read_excel(symbol_list_file)
    ticker_list = ticker_list[ticker_list['TR.RIC'].isnull()==False]
    ticker_list.sort_values('best_ric', inplace = True)
    ticker_list.drop_duplicates(subset = ['best_ric'], inplace = True)


ticker_list = ticker_list[ticker_list['TR.RegCountryCode'] == 'US']
ticker_list.reset_index(inplace = True)


def get_price_data(ric, sa_ticker):
    print("Getting price data for %s" % ric)
    
    fields = [
              ek.TR_Field('tr.priceclose.date'),
              ek.TR_Field('tr.priceclose.currency'),
              ek.TR_Field('tr.open'), 
              ek.TR_Field('tr.close'), 
              ek.TR_Field('tr.high'), 
              ek.TR_Field('tr.low'), 
              ek.TR_Field('tr.volume'), 
              ek.TR_Field('tr.value'), 
              ek.TR_Field('tr.count')
              ]
    data, err = ek.get_data([ric], fields, {'SDate': '2000-01-01', 'EDate': '2019-12-31', 'adjusted':'Y'})
    
    data = pd.concat([pd.Series(sa_ticker, index = data.index, name = 'sa_ticker'), 
                      data], axis = 1)
    
    data.set_index('Date', inplace = True)
    
    data.dropna(subset=['Price Open','Price Close'], inplace=True)
    
    return data


def get_dividend_data(ric, sa_ticker):
    print("Getting dividend for %s" % ric)
    
    fields = [
              ek.TR_Field('tr.DivPayDate'),  
              ek.TR_Field('tr.DivExDate'),
              ek.TR_Field('tr.DivTaxStatus'),
              ek.TR_Field('tr.DivUnadjustedNet'), 
              ek.TR_Field('tr.DivUnadjustedGross'), 
              ek.TR_Field('tr.DivUnadjustedGross.currency'),
              ek.TR_Field('tr.DivAdjustedNet'), 
              ek.TR_Field('tr.DivAdjustedGross'), 
              ek.TR_Field('tr.DivAdjustedGross.currency')
              ]
    data, err = ek.get_data([ric], fields, {'SDate': '2000-01-01', 'EDate': '2019-08-20', 'adjusted':'Y'})
    
    data = pd.concat([pd.Series(sa_ticker, index = data.index, name = 'sa_ticker'), 
                      data], axis = 1)
    
    return data



def get_event_data(ric, sa_ticker):
    print("Getting event data for %s" % ric)
    
    fields = [
              ek.TR_Field('TR.EventTitle.date'),  
              ek.TR_Field('TR.EventTitle.instrument'), 
              ek.TR_Field('TR.EventTitle.companyEvent'), 
              ek.TR_Field('TR.EventTitle.value')
              ]
    
    data, err = ek.get_data([ric], fields, {'SDate': '2015-01-01', 'EDate': '2019-08-20', 'EventType': 'ECALL;RES'})
    data2, err = ek.get_data([ric], fields, {'SDate': '2010-01-01', 'EDate': '2014-12-31', 'EventType': 'ECALL;RES'})
    data3, err = ek.get_data([ric], fields, {'SDate': '2005-01-01', 'EDate': '2009-12-31', 'EventType': 'ECALL;RES'})
    
    data = pd.concat([data, data2, data3])
    
    data = pd.concat([pd.Series(sa_ticker, index = data.index, name = 'sa_ticker'), 
                      data], axis = 1)
    
    return data


def get_eps_data(ric, sa_ticker):
    print("Getting EPS data for %s" % ric)
    
    fields = [
              ek.TR_Field('TR.EPSActValue.fperiod'),
              ek.TR_Field('TR.EPSActPeriodMon'),
              ek.TR_Field('TR.EPSActPeriodYr'),
              ek.TR_Field('TR.EPSActReportDate'),
              ek.TR_Field('TR.EPSActValue'), 
              ek.TR_Field('TR.EPSActValue.currency'), 
              ek.TR_Field('TR.EPSMean.date'),
              ek.TR_Field('TR.EPSMean'), 
              ek.TR_Field('TR.EPSMedian'), 
              ek.TR_Field('TR.EpsSmartEst'),
              ek.TR_Field('TR.EPSEstimateDiffusion'),
              ek.TR_Field('TR.EPSStdDev'), 
              ek.TR_Field('TR.EPSNumberOfEstimates'),
              ek.TR_Field('TR.EPSNumIncEstimates')
              ]
    data, err = ek.get_data([ric], fields, {'SDate': '2000-01-01', 'EDate': '2019-08-20', 
                                            'period': '1FQ', 'adjusted':'Y'})
    
    data = pd.concat([pd.Series(sa_ticker, index = data.index, name = 'sa_ticker'), 
                      data], axis = 1)
    
    data.dropna(subset=['Earnings Per Share - Actual', 'Currency'], inplace=True)
    
    return data


def get_divest_data(ric, sa_ticker):
    print("Getting dividend estimate data for %s" % ric)
    
    fields = [
              ek.TR_Field('TR.DPSActValue.fperiod'),
              ek.TR_Field('TR.DPSActPeriodMon'),
              ek.TR_Field('TR.DPSActPeriodYr'),
              ek.TR_Field('TR.DPSActReportDate'),
              ek.TR_Field('TR.DPSActValue'), 
              ek.TR_Field('TR.DPSActValue.currency'), 
              ek.TR_Field('TR.DPSMean.date'),
              ek.TR_Field('TR.DPSMean'), 
              ek.TR_Field('TR.DPSMedian'), 
              ek.TR_Field('TR.DPSStdDev'), 
              ek.TR_Field('TR.DPSNumberOfEstimates'),
              ek.TR_Field('TR.DPSNumIncEstimates')
              ]
    data, err = ek.get_data([ric], fields, {'SDate': '2000-01-01', 'EDate': '2019-08-20', 
                                            'period': '1FQ', 'adjusted':'Y'})
    
    data = pd.concat([pd.Series(sa_ticker, index = data.index, name = 'sa_ticker'), 
                      data], axis = 1)
    
    data.dropna(subset=['Period Year'], inplace=True)
    
    return data


def get_cfps_data(ric, sa_ticker):
    print("Getting cash flow data for %s" % ric)
    
    fields = [
              ek.TR_Field('TR.CFPSActValue.fperiod'),
              ek.TR_Field('TR.CFPSActPeriodMon'),
              ek.TR_Field('TR.CFPSActPeriodYr'),
              ek.TR_Field('TR.CFPSActReportDate'),
              ek.TR_Field('TR.CFPSActValue'), 
              ek.TR_Field('TR.CFPSActValue.currency'), 
              ek.TR_Field('TR.CFPSMean.date'),
              ek.TR_Field('TR.CFPSMean'), 
              ek.TR_Field('TR.CFPSMedian'), 
              ek.TR_Field('TR.CFPSStdDev'), 
              ek.TR_Field('TR.CFPSNumOfEstimates'),
              ek.TR_Field('TR.CFPSNumIncEstimates')
              ]
    data, err = ek.get_data([ric], fields, {'SDate': '2000-01-01', 'EDate': '2019-08-20', 
                                            'period': '1FQ', 'adjusted':'Y'})
    
    data = pd.concat([pd.Series(sa_ticker, index = data.index, name = 'sa_ticker'), 
                      data], axis = 1)
    
    data.dropna(subset=['Period Year'], inplace=True)
    
    return data


def get_fcfps_data(ric, sa_ticker):
    print("Getting free cash flow data for %s" % ric)
    
    fields = [
              ek.TR_Field('TR.FCFPSActValue.fperiod'),
              ek.TR_Field('TR.FCFPSActPeriodMon'),
              ek.TR_Field('TR.FCFPSActPeriodYr'),
              ek.TR_Field('TR.FCFPSActReportDate'),
              ek.TR_Field('TR.FCFPSActValue'), 
              ek.TR_Field('TR.FCFPSActValue.currency'), 
              ek.TR_Field('TR.FCFPSMean.date'),
              ek.TR_Field('TR.FCFPSMean'), 
              ek.TR_Field('TR.FCFPSMedian'), 
              ek.TR_Field('TR.FCFPSStdDev'), 
              ek.TR_Field('TR.FCFPSNumOfEstimates'),
              ek.TR_Field('TR.FCFPSNumIncEstimates'),
              ]
    data, err = ek.get_data([ric], fields, {'SDate': '2000-01-01', 'EDate': '2019-08-20', 
                                            'period': '1FQ', 'adjusted':'Y'})
    
    data = pd.concat([pd.Series(sa_ticker, index = data.index, name = 'sa_ticker'), 
                      data], axis = 1)
    
    data.dropna(subset=['Period Year'], inplace=True)
    
    return data


def get_roe_data(ric, sa_ticker):
    print("Getting ROE data for %s" % ric)
    
    fields = [
              ek.TR_Field('TR.ROEActValue.fperiod'),
              ek.TR_Field('TR.ROEActPeriodMon'),
              ek.TR_Field('TR.ROEActPeriodYr'),
              ek.TR_Field('TR.ROEActReportDate'),
              ek.TR_Field('TR.ROEActValue'), 
              ek.TR_Field('TR.ROEMean.date'),
              ek.TR_Field('TR.ROEMean'), 
              ek.TR_Field('TR.ROEMedian'), 
              ek.TR_Field('TR.ROEStdDev'), 
              ek.TR_Field('TR.ROENumOfEstimates'),
              ek.TR_Field('TR.ROENumIncEstimates'),
              ]
    data, err = ek.get_data([ric], fields, {'SDate': '2000-01-01', 'EDate': '2019-08-20', 
                                            'period': '1FQ', 'adjusted':'Y'})
    
    data = pd.concat([pd.Series(sa_ticker, index = data.index, name = 'sa_ticker'), 
                      data], axis = 1)
    
    data.dropna(subset=['Period Year'], inplace=True)
    
    return data


def get_roa_data(ric, sa_ticker):
    print("Getting ROA data for %s" % ric)
    
    fields = [
              ek.TR_Field('TR.ROAActValue.fperiod'),
              ek.TR_Field('TR.ROAActPeriodMon'),
              ek.TR_Field('TR.ROAActPeriodYr'),
              ek.TR_Field('TR.ROAActReportDate'),
              ek.TR_Field('TR.ROAActValue'), 
              ek.TR_Field('TR.ROAMean.date'),
              ek.TR_Field('TR.ROAMean'), 
              ek.TR_Field('TR.ROAMedian'), 
              ek.TR_Field('TR.ROAStdDev'), 
              ek.TR_Field('TR.ROANumOfEstimates'),
              ek.TR_Field('TR.ROANumIncEstimates')
              ]
    data, err = ek.get_data([ric], fields, {'SDate': '2000-01-01', 'EDate': '2019-08-20', 
                                            'period': '1FQ', 'adjusted':'Y'})
    
    data = pd.concat([pd.Series(sa_ticker, index = data.index, name = 'sa_ticker'), 
                      data], axis = 1)
    
    data.dropna(subset=['Period Year'], inplace=True)
    
    return data


def get_revenue_data(ric, sa_ticker):
    print("Getting revenue data for %s" % ric)
    
    fields = [
              ek.TR_Field('TR.RevenueActValue.fperiod'),
              ek.TR_Field('TR.RevenueActPeriodMon'),
              ek.TR_Field('TR.RevenueActPeriodYr'),
              ek.TR_Field('TR.RevenueActReportDate'),
              ek.TR_Field('TR.RevenueActValue'), 
              ek.TR_Field('TR.RevenueActValue.currency'), 
              ek.TR_Field('TR.RevenueMean.date'),  
              ek.TR_Field('TR.RevenueMean'), 
              ek.TR_Field('TR.RevenueMedian'), 
              ek.TR_Field('TR.RevenueEstimateDiffusion'),
              ek.TR_Field('TR.RevenueStdDev'), 
              ek.TR_Field('TR.RevenueNumOfEstimates'),
              ek.TR_Field('TR.RevenueNumIncEstimates')
              ]
    data, err = ek.get_data([ric], fields, {'SDate': '2000-01-01', 'EDate': '2019-08-20', 
                                            'period': '1FQ', 'adjusted':'Y'})
    
    data = pd.concat([pd.Series(sa_ticker, index = data.index, name = 'sa_ticker'), 
                      data], axis = 1)
    
    data.dropna(subset=['Revenue - Actual', 'Currency'], inplace=True)
    
    return data


def get_market_cap_data(ric, sa_ticker):
    print("Getting market cap data for %s" % ric)
    
    fields = [
              ek.TR_Field('TR.CompanyMarketCap.date'),
              ek.TR_Field('TR.CompanyMarketCap.currency'),
              ek.TR_Field('TR.CompanyMarketCap'),
              #ek.TR_Field('TR.CompanySharesOutstanding'), # somehow not working with this field
              ek.TR_Field('TR.SharesOutstanding'),
              ek.TR_Field('TR.PriceToBVPerShare'),
              ek.TR_Field('TR.Volume'),
              ek.TR_Field('TR.Close')
              ]
    data, err = ek.get_data([ric], fields, {'SDate': '2000-01-01', 'EDate': '2019-12-31', 'adjusted':'Y'})
    
    data = pd.concat([pd.Series(sa_ticker, index = data.index, name = 'sa_ticker'), 
                      data], axis = 1)
    
    data.set_index('Date', inplace = True)
    
    data.dropna(subset=['Company Market Cap'], inplace=True)
    
    return data


def get_bvps_data(ric, sa_ticker):
    print("Getting BVPS data for %s" % ric)
    
    fields = [
              ek.TR_Field('TR.BVPSActValue.fperiod'),
              ek.TR_Field('TR.BVPSActPeriodMon'),
              ek.TR_Field('TR.BVPSActPeriodYr'),
              ek.TR_Field('TR.BVPSActReportDate'),
              ek.TR_Field('TR.BVPSActValue'), 
              ek.TR_Field('TR.BVPSActValue.currency'), 
              ek.TR_Field('TR.BVPSMean.date'),  
              ek.TR_Field('TR.BVPSMean'), 
              ek.TR_Field('TR.BVPSMedian'), 
              ek.TR_Field('TR.BVPSEstimateDiffusion'),
              ek.TR_Field('TR.BVPSStdDev'), 
              ek.TR_Field('TR.BVPSNumOfEstimates'),
              ek.TR_Field('TR.BVPSNumIncEstimates')
              ]
    data, err = ek.get_data([ric], fields, {'SDate': '2000-01-01', 'EDate': '2019-08-20', 
                                            'period': '1FQ', 'adjusted':'Y'})
    
    data = pd.concat([pd.Series(sa_ticker, index = data.index, name = 'sa_ticker'), 
                      data], axis = 1)
    
    data.dropna(subset=['Book Value Per Share - Actual', 'Currency'], inplace=True)
    
    return data



def process_ticker(row, ticker):
    print("Processing ticker %s of %d" % (row, len(ticker_list)))
    
    sa_ticker = ticker['sa_ticker']
    ric = ticker['best_ric']
    
    if ric is None or ric is np.nan:
        print("Skipping")
        return
    
    price_out_file = out_dir + ric + '_price.csv'
    dividend_out_file = out_dir + ric + '_div.csv'
    eps_out_file = out_dir + ric + '_eps.csv'
    rev_out_file = out_dir + ric + '_revenue.csv'
    mkt_out_file = out_dir + ric + '_marketcap.csv'
    bvps_out_file = out_dir + ric + '_bvps.csv'
    cfps_out_file = out_dir + ric + '_cfps.csv'
    fcfps_out_file = out_dir + ric + '_fcfps.csv'
    roe_out_file = out_dir + ric + '_roe.csv'
    roa_out_file = out_dir + ric + '_roa.csv'
    divest_out_file = out_dir + ric + '_divest.csv'
    events_out_file = out_dir + ric + '_event.csv'
    
    down = False
    
    try:
        if not os.path.isfile(price_out_file) and 'PRICE' in DATA_ITEMS:
            price_data = get_price_data(ric, sa_ticker)
            price_data.to_csv(price_out_file)
            down = True
    except KeyError as e:
        print("\tError retrieving price data (KeyError)!")
        pd.DataFrame().to_csv(price_out_file)
    except Exception as e:
        print("\tError retrieving price data!")
        
    try:
        if not os.path.isfile(dividend_out_file) and 'DIVIDEND' in DATA_ITEMS:
            dividend_data = get_dividend_data(ric, sa_ticker)
            dividend_data.to_csv(dividend_out_file)
            down = True
    except KeyError as e:
        print("\tError retrieving dividend data (KeyError)!")
        pd.DataFrame().to_csv(dividend_out_file)
    except Exception as e:
        print("\tError retrieving dividend data!")
    
    try:
        if not os.path.isfile(events_out_file) and 'EVENT' in DATA_ITEMS:
            event_data = get_event_data(ric, sa_ticker)
            event_data.to_csv(events_out_file)
            down = True
    except KeyError as e:
        print("\tError retrieving event data (KeyError)!")
        pd.DataFrame().to_csv(events_out_file)
    except Exception as e:
        print("\tError retrieving event data!")
    
    try:
        if not os.path.isfile(eps_out_file) and 'EPS' in DATA_ITEMS:
            eps_data = get_eps_data(ric, sa_ticker)
            eps_data.to_csv(eps_out_file)
            down = True
    except KeyError as e:
        print("\tError retrieving EPS data (KeyError)!")
        pd.DataFrame().to_csv(eps_out_file)
    except Exception as e:
        print("\tError retrieving EPS data!")
    
    try:
        if not os.path.isfile(rev_out_file) and 'REVENUE' in DATA_ITEMS:
            rev_data = get_revenue_data(ric, sa_ticker)
            rev_data.to_csv(rev_out_file)
            down = True
    except KeyError as e:
        print("\tError retrieving revenue data (KeyError)!")
        pd.DataFrame().to_csv(rev_out_file)
    except Exception as e:
        print("\tError retrieving revenue data!")
    
    try:
        if not os.path.isfile(mkt_out_file) and 'MKT_CAP' in DATA_ITEMS:
            mkt_data = get_market_cap_data(ric, sa_ticker)
            mkt_data.to_csv(mkt_out_file)
            down = True
    except KeyError as e:
        print("\tError retrieving market cap data (KeyError)!")
        pd.DataFrame().to_csv(mkt_out_file)
    except Exception as e:
        print("\tError retrieving market cap data!")
    
    try:
        if not os.path.isfile(bvps_out_file) and 'BVPS' in DATA_ITEMS:
            bvps_data = get_bvps_data(ric, sa_ticker)
            bvps_data.to_csv(bvps_out_file)
            down = True
    except KeyError as e:
        print("\tError retrieving BVPS data (KeyError)!")
        pd.DataFrame().to_csv(bvps_out_file)
    except Exception as e:
        print("\tError retrieving BVPS data!")
    
    try:
        if not os.path.isfile(cfps_out_file) and 'CFPS' in DATA_ITEMS:
            cfps_data = get_cfps_data(ric, sa_ticker)
            cfps_data.to_csv(cfps_out_file)
            down = True
    except KeyError as e:
        print("\tError retrieving CFPS data (KeyError)!")
        pd.DataFrame().to_csv(cfps_out_file)
    except Exception as e:
        print("\tError retrieving CFPS data!")
    
    try:
        if not os.path.isfile(fcfps_out_file) and 'CFPS' in DATA_ITEMS:
            fcfps_data = get_fcfps_data(ric, sa_ticker)
            fcfps_data.to_csv(fcfps_out_file)
            down = True
    except KeyError as e:
        print("\tError retrieving FCFPS data (KeyError)!")
        pd.DataFrame().to_csv(fcfps_out_file)
    except Exception as e:
        print("\tError retrieving FCFPS data!")
    
    try:
        if not os.path.isfile(roe_out_file) and 'ROEA' in DATA_ITEMS:
            roe_data = get_roe_data(ric, sa_ticker)
            roe_data.to_csv(roe_out_file)
            down = True
    except KeyError as e:
        print("\tError retrieving ROE data (KeyError)!")
        pd.DataFrame().to_csv(roe_out_file)
    except Exception as e:
        print("\tError retrieving ROE data!")
    
    try:
        if not os.path.isfile(roa_out_file) and 'ROEA' in DATA_ITEMS:
            roa_data = get_roa_data(ric, sa_ticker)
            roa_data.to_csv(roa_out_file)
            down = True
    except KeyError as e:
        print("\tError retrieving ROA data (KeyError)!")
        pd.DataFrame().to_csv(roa_out_file)
    except Exception as e:
        print("\tError retrieving ROA data!")
        
    try:
        if not os.path.isfile(divest_out_file) and 'DIVEST' in DATA_ITEMS:
            divest_data = get_divest_data(ric, sa_ticker)
            divest_data.to_csv(divest_out_file)
            down = True
    except KeyError as e:
        print("\tError retrieving dividend estimate data (KeyError)!")
        pd.DataFrame().to_csv(divest_out_file)
    except Exception as e:
        print("\tError retrieving dividend estimate data!")
    
    if down:
        time.sleep(row % 5 + 1)



#for row, ticker in ticker_list.iterrows():
#    process_ticker(row, ticker)

from multiprocessing.pool import ThreadPool

with ThreadPool(3) as pool:
    results = pool.starmap(process_ticker, ticker_list.iterrows())



