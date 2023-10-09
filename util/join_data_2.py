# -*- coding: utf-8 -*-
"""
Created on Sat May  6 18:58:13 2023

@author: aq75iwit
"""
import gc
import random
import time
import json
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
import zipline as zl
import numpy as np
import re
import os
import multiprocessing as mp
import pickle

from sklearn.linear_model import LinearRegression

from config import TMP_FOLDER, CALL_FILE, EXPORT_FOLDER, INDICES
from util.add_features import add_sentiment_features
from util.helper import get_last_period

def get_joined_call_data(call_file=CALL_FILE, 
                         export_folder=EXPORT_FOLDER, 
                         indices=INDICES, 
                         use_sa_only=True):
    print("Getting joined call data ...")
    
    base_filename = os.path.split(call_file)[1]
    tmp_file = TMP_FOLDER + base_filename + '.pkl'
    
    if not os.path.isfile(tmp_file):
        read_and_join_data(call_file, export_folder, indices, tmp_file, use_sa_only)
#        with mp.Pool(processes = 1) as pool:
#            pool.apply(read_and_join_data, 
#                       (call_file, export_folder, indices, tmp_file, use_sa_only))
    
    with open(tmp_file, 'rb') as f:
        data = pickle.load(f)
    
    return data


def read_call_data(call_file):
    with open(call_file, 'r') as f:
        json_data = json.load(f)
    
    data = pd.DataFrame(json_data)
    
    #data.drop(['general', 'qanda'], axis = 1, inplace = True)
    
    json_data = None
    
    # Fix to convert NaN string to actual null
    data.loc[data.ticker_symbol == 'NaN', 'ticker_symbol'] = np.nan
    data.loc[data.quarter == 'NaN', 'quarter'] = np.nan
    data.loc[data.year == 'NaN', 'year'] = np.nan
    data.loc[data.iso_date == 'NaN', 'iso_date'] = np.nan
    
    # Fix to add 'Z' to timestamp
    data['iso_date'] = data.iso_date.apply(lambda x: 
        x if (pd.isna(x) or x.endswith('Z')) else x + 'Z')
    
    print("Found %d calls with no ticker!" % data.ticker_symbol.isna().sum())
    data.dropna(subset=['ticker_symbol'], axis=0, inplace=True)
    
    gc.collect()
    
    return data

RETURN_WINDOWS = [#(0, 0, 'call_day'), 
                  #(1, 1, 'next_day'),
                  *[(2, d, '+2_+%d_bday' % d) for d in [2, 3, 4, 5, 7, 9, 10, \
                      15, 20, 25, 30, 35, 40, 45, 50, 55, 60]],
                  *[(-d, 0, '-%d_+0_bday' % d) for d in [0, 1, 2, 3, 4, 5]]
                  ]
    
def get_price_data(export_folder, ticker_list):
    prices = pd.read_hdf('{}adjusted_prices.hdf'.format(export_folder))

    prices = prices[['instrument_key', 'Date', 'div', 'Open', 'Close', 'Price Close']]
    # prices = prices.head(50000)

    prices = prices[prices.Date >= '2007-01-01']
    instr_ids = list(ticker_list) + ['SP1500ETF']
    prices = prices[prices.instrument_key.isin(instr_ids)]

    prices.loc[prices.Open.abs() > 1E6, 'Open'] = np.nan
    prices.loc[prices.Close.abs() > 1E6, 'Close'] = np.nan
    prices.loc[prices['div'].abs() > 1E6, 'div'] = np.nan

    prices.sort_values('Date', inplace=True)
    prices.set_index(['Date'], inplace=True)

    return prices


def join_fama_french_expected_returns(row, prices, fama_data, cal):
    print(row.name)

    regr_offset = CustomBusinessDay(6, 
                                    holidays=cal.adhoc_holidays, 
                                    calendar=cal.regular_holidays)

    dt = row.final_datetime

    stock_prices = prices.loc[prices.instrument_key == row.ticker_symbol, 'Close']
    stock_returns = stock_prices.pct_change()

    last_regression_datetime = dt - regr_offset

    fama_data_block = fama_data.loc[:last_regression_datetime].tail(250)

    reg_data = fama_data_block.join(stock_returns).dropna()

    if len(reg_data) == 0:
        return pd.Series()

    regr = LinearRegression(n_jobs=-1)
    regr.fit(reg_data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']],
             reg_data.Close - reg_data.RF)

    result = {
        'fama_french_a': regr.intercept_,
        'fama_french_b': regr.coef_[0],
        'fama_french_s': regr.coef_[1],
        'fama_french_h': regr.coef_[2],
        'fama_french_r': regr.coef_[3],
        'fama_french_c': regr.coef_[4]
    }

    pred_data = fama_data.loc[dt:].head(300)
    exp_ret = regr.predict(pred_data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]) +\
        pred_data.RF

    exp_ret.name = 'exp_return'

    exp_ret = pd.DataFrame(exp_ret).join(stock_returns)

    for start_offset, end_offset, name in RETURN_WINDOWS:
        start = dt + CustomBusinessDay(start_offset,
                                       holidays=cal.adhoc_holidays, calendar=cal.regular_holidays)
        end = dt + CustomBusinessDay(end_offset,
                                     holidays=cal.adhoc_holidays, calendar=cal.regular_holidays)
        rets = exp_ret[start:end]

        result['exp_prod_return_' + name] = (rets.exp_return + 1).prod() - 1
        result['exp_sum_return_' + name] = rets.exp_return.sum()
    
        abn = rets.Close - rets.exp_return
    
        result['ff5_abnormal_sum_return_' + name] = abn.sum()
        result['ff5_abnormal_prod_return_' + name] = (abn + 1).prod() - 1
    
    return pd.Series(result)

def apply_fama_french_on_block(df, export_folder):
    time.sleep(random.randrange(0, 60))

    fama_data = pd.read_csv(export_folder +'/F-F_Research_Data_5_Factors_2x3_daily.CSV', skiprows=3)
    fama_data.set_index(pd.to_datetime(fama_data['Unnamed: 0'], format='%Y%m%d', utc=True),
                        inplace=True)
    fama_data = fama_data['2005-01-01':]

    fama_data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']] /= 100

    prices = get_price_data(export_folder, list(df.ticker_symbol.unique()))

    cal = zl.get_calendar('NYSE')

    result = df.apply(join_fama_french_expected_returns, axis=1,
                      args=(prices, fama_data, cal))

    return result


def join_fama_french_decile_returns(row, ff_dec_ret, cal):
    if pd.isna(row['FF ME Quantile']):
        return pd.Series()

    exp_ret = ff_dec_ret[row['FF Size Decile']]
    dt = row.final_datetime

    result = {}

    for start_offset, end_offset, name in RETURN_WINDOWS:
        start = dt + CustomBusinessDay(start_offset,
                                       holidays=cal.adhoc_holidays, calendar=cal.regular_holidays)
        end = dt + CustomBusinessDay(end_offset,
                                     holidays=cal.adhoc_holidays, calendar=cal.regular_holidays)

        rets = exp_ret[start:end]

        result['decile_exp_prod_return_' + name] = (rets + 1).prod() - 1
        result['decile_exp_sum_return_' + name] = rets.sum()

        # abn = rets.Close - rets.exp_return
        # result['ff5_abnormal_sum_return_' + name] = abn.sum()
        # result['ff5_abnormal_prod_return_' + name] = (abn + 1).prod() - 1

    return pd.Series(result)

def apply_fama_french_decile_on_block(df, export_folder):
    time.sleep(random.randrange(0, 60))

    # use equal weight returns -> skip first rows with market weight returns
    ff_dec_ret = pd.read_csv(export_folder +'F-F_Portfolios_Formed_on_ME_daily.CSV',
                             skiprows=24618).dropna()
    ff_dec_ret.set_index(pd.to_datetime(ff_dec_ret['Date'],
                                        format="%Y%m%d"),  # .dt.strftime('%Y-%m-%d'),
                         inplace=True)
    ff_dec_ret = ff_dec_ret.loc['2000-01-03':] / 100

    cal = zl.get_calendar('NYSE')

    result = df.apply(join_fama_french_decile_returns, axis=1,
                      args=(ff_dec_ret, cal))

    return result


def read_and_join_data(call_file, export_folder, indices, tmp_file,
                       use_sa_only=True):

    cal = zl.get_calendar('NYSE')

    print("Reading call data ...")
    data = read_call_data(call_file)

    # data = pd.read_hdf('/home/mschnaubelt/tmp/earnings_calls/meta_debug.hdf')
    # data = data[['company_name', 'filename', 'iso_date', 'length',
    #             'perspective', 'quarter', 'release', 'release_time',
    #             'ticker_symbol', 'year', 'datetime']]

    sa_id_pat = re.compile('^(([0-9]+))', re.IGNORECASE)
    tr_id_pat = re.compile('(([0-9]+)-Transcript)', re.IGNORECASE)

    def file_id(p):
        if '.txt' in p:
            m = re.search(tr_id_pat, p)
            mtype = 'TR'
        else:
            m = re.search(sa_id_pat, p)
            mtype = 'SA'

        if m is None or len(m.groups()) != 2:
            return np.nan

        return mtype + str(int(m.groups()[1]))


def fiscal_period(r):
    if not (pd.isna(r['year']) or pd.isna(r['quarter'])):
        return f'FY{r["year"]:04d}Q{r["quarter"]}'
    else:
        return None

data['sa_fiscal_period'] = data.apply(fiscal_period, axis=1)


### prepare date column in ET time zone ###

datetime = pd.to_datetime(data.iso_date, errors='coerce', utc=True)

parsed_datetime = pd.to_datetime(data.release.str.replace('Transcript', '')
                                 .str.replace('Session', ''),
                                 errors='coerce', utc=True)

datetime.loc[datetime.isna()] = parsed_datetime

data['release_datetime'] = datetime

data['local_date'] = datetime.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
data.loc[data.local_date == 'NaT', 'local_date'] = np.nan

print(f"Found {data.local_date.isna().sum()} calls with missing release date information")


### join index data to get link between SA ticker symbol and Eikon RIC/Instrument Code ###

index = pd.read_hdf(f'{export_folder}ticker_index.hdf')

index.set_index('sa_ticker', inplace=True, drop=False)
data = data.join(index[['rics', 'instrument_id', 'primary_ric', 'country',
                        'naics', 'naics_sec', 'naics_subsec',
                        'trbc', 'trbc_sec', 'type_code']],
                 on='ticker_symbol')

print(f"Total of {data.primary_ric.isna().sum()} calls could not be matched to Eikon instruments")

print(f"Total of {len(data[data.primary_ric.isna()].ticker_symbol.unique())} SA tickers could not be matched to Eikon instruments")

mdata = data[data.primary_ric.isna()]
mdata['company_name'] = mdata['company_name'].str.replace('TRANSCRIPT SPONSOR', '').str.strip()
mdata = mdata.groupby('ticker_symbol').aggregate({
    'company_name': ['first', 'count'],
    'file_id': ['first', 'last']}).sort_values(('company_name', 'count'), ascending=False)
mdata.to_excel('missing_index.xlsx')


# join event data (earnings calls only) ###
print("Joining events data ...")

events = pd.read_hdf(export_folder + 'events.hdf')
    
events.columns = ['instrument_key', 'Date', 'eikon_event_type', 'eikon_event_title', 
                  'eikon_f_year', 'eikon_f_quarter', 'eikon_fiscal_period']
events = events[events.eikon_event_type == 'EarningsCallsAndPresentations']
events = events[events.eikon_event_title.str.contains('Earnings', flags = re.IGNORECASE) \
                & events.eikon_event_title.str.contains('Call', flags = re.IGNORECASE)]

events.dropna(subset = ['eikon_fiscal_period'], inplace = True)

events['local_date'] = events.Date.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')

events.sort_values('Date', inplace = True)
events.drop_duplicates(subset = ['instrument_key', 'local_date'], inplace = True)

events.set_index(['instrument_key', 'local_date'], inplace = True, drop = False)


# first, join events by call date
data = data.join(events[['eikon_event_title', 'eikon_f_year', 
                         'eikon_f_quarter', 'eikon_fiscal_period']], 
                 on = ['ticker_symbol', 'local_date'],
                 how = 'left' if use_sa_only else 'outer', rsuffix = ' by date')

if not use_sa_only:
    data.drop(columns = ['rics', 'instrument_id', 'primary_ric'], inplace = True)
    data = data.join(index[['rics', 'instrument_id', 'primary_ric']], 
                     on = 'ticker_symbol')

# second, join events by fiscal period
events.set_index(['instrument_key', 'eikon_fiscal_period'], inplace = True, drop = False)

data = data.join(events[['eikon_event_title', 'eikon_f_year', 
                  'eikon_f_quarter', 'local_date']], 
                 on = ['ticker_symbol', 'sa_fiscal_period'],
                 how = 'left', rsuffix = ' by FP')

sa_date_corrupted = (data['local_date by FP'] != data.local_date) \
                        & ~data['local_date by FP'].isna()\
                        & data.eikon_fiscal_period.isna()\
                        & data.local_date.isna()

data.loc[sa_date_corrupted, 'local_date'] = data.loc[sa_date_corrupted, 'local_date by FP']


# do sanity checks of SA data
data.loc[(data.release_datetime.dt.year - data.year).abs() > 1, 'sa_corrupt'] = True
data['sa_corrupt'].fillna(False, inplace = True)


# determine fiscal period from SA and Eikon data
fiscal_period_mismatches = (data['eikon_fiscal_period'] != data.sa_fiscal_period) & \
                                (data['eikon_fiscal_period'].isna() == False) & \
                                (data['sa_fiscal_period'].isna() == False)


# fill in Eikon data if period mismatch or SA info not there and SA local date seems OK
data['fiscal_period'] = data['sa_fiscal_period']
eikon_fp_ok = (fiscal_period_mismatches | data.sa_fiscal_period.isna()) & (~data.sa_corrupt)
data.loc[eikon_fp_ok, 'fiscal_period'] = data.loc[eikon_fp_ok, 'eikon_fiscal_period']


missing_period = data.fiscal_period.isna() & data.filename.str.contains('earnings')
print("Found %d calls with missing period information" % missing_period.sum())

data[missing_period].to_csv('missing_financial_period.csv')

# find calls where ticker symbol might be corrupt
def find_ticker_in_filename(r):
    fn = r['filename']
    t = r['ticker_symbol']
    return t.lower() in fn.lower()

#data['ticker_match'] = data.apply(find_ticker_in_filename, axis = 1)


# based on final local_date and fiscal_period entries, drop duplicates

data.dropna(subset = ['local_date'], inplace = True)
data.dropna(subset = ['fiscal_period'], inplace = True)

dups = data.duplicated(subset = ['ticker_symbol', 'local_date', 'fiscal_period'], 
                       keep = False)
print("Found %d duplicates" % dups.sum())
data[dups].to_csv('duplicates.csv')

#data.drop(dups[dups].index, inplace = True)
data.sort_values('file_id', inplace = True)
data.drop_duplicates(subset = ['ticker_symbol', 'local_date', 'fiscal_period'],
                     keep = 'last', inplace = True)



# add last year/quarter fiscal periods 
data['last_year_fiscal_period'] = data['fiscal_period'].apply(get_last_period, mode = 'Y')
data['last_quarter_fiscal_period'] = data['fiscal_period'].apply(get_last_period, mode = 'Q')


# prepare EPS data
print("Joining EPS data ...")

eps = pd.read_hdf(export_folder + 'eps.hdf')

# join current EPS data
eps_to_merge = eps.set_index(['instrument_key', 'Financial Period Absolute'])[[
        'EPS Actual', 'EPS Mean Estimate', 
        'EPS Median Estimate',  'EPS Estimate Standard Deviation', 
        'EPS Number of Estimates', 'EPS Number of Included Estimates', 
        'EPS Smart Estimate', 'Report Date', 
        'EPS Mean Estimate min', 'EPS Mean Estimate max', 'EPS Mean Estimate first',
        'EPS Mean Estimate std', 'EPS Mean Estimate count', 'EPS Mean Estimate unique_count', 
        'Date span', 'EPS Smart Estimate min', 'EPS Smart Estimate max', 
        'EPS Smart Estimate first', 'EPS Smart Estimate std', 'EPS Smart Estimate count',
        'EPS Smart Estimate unique_count', 'EPS Forward Estimate']]

eps_to_merge = eps_to_merge.loc[~ eps_to_merge.index.duplicated()]

eps_to_merge.rename(columns = {'Report Date': 'EPS Report Date'}, inplace = True)

data = data.join(eps_to_merge, on = ['ticker_symbol', 'fiscal_period'])

# join last year/quarter eps
eps_to_merge = eps.set_index(['instrument_key', 'Financial Period Absolute'])[[
        'EPS Actual', 'EPS Mean Estimate']]
eps_to_merge = eps_to_merge.loc[~ eps_to_merge.index.duplicated()]

eps_to_merge.columns = ['EPS Actual Previous Year', 'EPS Mean Estimate Previous Year']
data = data.join(eps_to_merge, on = ['ticker_symbol', 'last_year_fiscal_period'])

eps_to_merge.columns = ['EPS Actual Previous Quarter', 'EPS Mean Estimate Previous Quarter']
data = data.join(eps_to_merge, on = ['ticker_symbol', 'last_quarter_fiscal_period'])


# calculate historic stdev and drift of actual EPS
seps = eps.set_index(['instrument_key', 'Financial Period Absolute']).sort_index()

roll_mean = seps.groupby(level = 0).rolling(4, min_periods = 4)[
                'EPS Actual'].mean().reset_index(level = 0, drop = True)
roll_std = seps.groupby(level = 0).rolling(4, min_periods = 4)[
                'EPS Actual'].std().reset_index(level = 0, drop = True)
roll_drift = seps.groupby(level = 0)['EPS Actual'].diff()\
                .groupby(level = 0).rolling(4, min_periods = 4).mean().reset_index(level = 0, drop = True)

roll_mean.name = 'EPS Actual Historic Mean'
roll_std.name = 'EPS Actual Historic StD'
roll_drift.name = 'EPS Actual Historic Drift'

data = data.join(roll_mean, on = ['ticker_symbol', 'fiscal_period'])
data = data.join(roll_std, on = ['ticker_symbol', 'fiscal_period'])
data = data.join(roll_drift, on = ['ticker_symbol', 'fiscal_period'])



# join other fundamental data
FUND_TYPES = ['REVENUE', 'BVPS', 'CFPS', 'FCFPS', 'DIVEST', 'ROA', 'ROE', 
              'CFO', 'FCF', 'GPM', 'NETDEBT', 'NETINC', 'OPEX', 'PRETAXINC', 
              'TOTASSETS', 'OPROFIT', 'SHEQUITY', 'TOTDEBT']
for f_type in FUND_TYPES:
    print("Joining %s data ..." % f_type)
    fund_data = pd.read_hdf(export_folder + '%s.hdf' % f_type.lower())
    
    cols = ['Actual', 'Estimate Mean', 'Estimate StD', 
            'NEstimates', 'NIncEstimates', 'Forward Estimate']
    cols = [f_type + ' ' + c for c in cols] + ['Report Date']
    
    cols = list(filter(lambda c: c in fund_data.columns, cols))
    
    fund_data_to_merge = fund_data.set_index(['instrument_key', 'Financial Period Absolute'])[cols]
    fund_data_to_merge = fund_data_to_merge.loc[~ fund_data_to_merge.index.duplicated()]
    
    fund_data_to_merge.rename(columns = {'Report Date': f_type + ' Report Date'}, inplace = True)
    
    s_fund_data = fund_data_to_merge.sort_index()
    roll_mean = s_fund_data.groupby(level = 0).rolling(4, min_periods = 4)[
                f_type + ' Actual'].mean().reset_index(level = 0, drop = True)
    roll_mean.name = f_type + ' Actual 4Q Mean'
    
    lq_fund_data = fund_data_to_merge[[f_type + ' Actual']]
    lq_fund_data.columns = [f_type + ' Actual Previous Quarter']
    
    data = data.join(fund_data_to_merge, on = ['ticker_symbol', 'fiscal_period'])
    data = data.join(roll_mean, on = ['ticker_symbol', 'fiscal_period'])
    data = data.join(lq_fund_data, on = ['ticker_symbol', 'last_quarter_fiscal_period'])



# join index data
print("Joining index data ...")

data['mkt_index'] = np.nan
all_index_instruments = []
all_index_rics = []
for index_name, index_file in indices.items():
    print("Joining index %s" % index_name)
    
    with open(export_folder + index_file, 'r') as f:
        index_data = json.load(f)
    
    index_data = pd.DataFrame(index_data)
    index_data['Date'] = pd.to_datetime(index_data.Date, utc = True, errors = 'coerce')
    
    index_data = index_data[index_data['Date'] >= '2005-01-01']
    
    index_data.set_index('Date', inplace = True)
    index_data = index_data.resample('1d').bfill()
    
    instruments_index = {}
    rics_index = {}
    for date, value in index_data.iterrows():
        ldate = date.tz_convert('America/New_York').strftime('%Y-%m-%d')
        
        instr_list = value['InstrumentIDs']
        for instr in instr_list:
            instruments_index[(ldate, instr)] = True
        
        ric_list = value['RICs']
        for ric in ric_list:
            rics_index[(ldate, ric)] = True
    
    all_index_instruments += [k[1] for k in instruments_index.keys()]
    all_index_rics += [k[1] for k in rics_index.keys()]
    
    def in_index(r):
        ld = r['local_date']
        
        rics = [r['primary_ric']]
        if r['rics'] is not np.nan:
            rics += r['rics']
            
        instr = r['instrument_id']
        
        in_ric = any([((ld, ric) in rics_index) for ric in rics])
        in_instr = (ld, instr) in instruments_index
        
        return in_ric or in_instr
    
    mkt_index = data.apply(in_index, axis = 1)
    data.loc[mkt_index, 'mkt_index'] = index_name


all_index_instruments = list(set(all_index_instruments))
all_index_rics = list(set(all_index_rics))


data.loc[data.mkt_index.isna(), 'mkt_index'] = 'NONE:' + \
                            data.loc[data.mkt_index.isna(), 'country']



### join price data ###
print("Joining price data ...")

prices = get_price_data(export_folder, list(data.ticker_symbol.unique()))


bfill_prices = prices.groupby('instrument_key').resample('1d').bfill(5)
bfill_prices.reset_index(level = 1, inplace = True)

bfill_prices['local_date'] = bfill_prices.Date.dt.strftime('%Y-%m-%d')
bfill_prices.set_index(['instrument_key', 'local_date'], inplace = True)

data['final_datetime'] = pd.to_datetime(data.local_date)


def add_prices(data, delta_days, delta_name, prices, use_index = False):
    days_offset = CustomBusinessDay(delta_days, 
                                    holidays = cal.adhoc_holidays, 
                                    calendar = cal.regular_holidays)
    
    join_datetime = data['final_datetime'] + days_offset
    join_symbol = pd.Series('SP1500ETF', index = data.index) if use_index else 'ticker_symbol'
    
    data = data.join(prices[['Open', 'Close']], 
                     on = [join_symbol, 
                           join_datetime.dt.strftime('%Y-%m-%d')],
                     rsuffix = delta_name)
    
    return data


data = add_prices(data, 0, '_call_day', bfill_prices, use_index = False)
data = add_prices(data, 1, '_next_bday', bfill_prices, use_index = False)

data = add_prices(data, 0, '_market_call_day', bfill_prices, use_index = True)
data = add_prices(data, 1, '_market_next_bday', bfill_prices, use_index = True)

for n in [1, 2, 3, 4, 5, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 250]:
    data = add_prices(data, n, '_+%d_bday' % n, bfill_prices, use_index = False)
    data = add_prices(data, n, '_market_+%d_bday' % n, bfill_prices, use_index = True)


ffill_prices = prices.groupby('instrument_key').resample('1d').ffill(5)
ffill_prices.reset_index(level = 1, inplace = True)

ffill_prices['local_date'] = ffill_prices.Date.dt.strftime('%Y-%m-%d')
ffill_prices.set_index(['instrument_key', 'local_date'], inplace = True)

for n in [-1, -2, -3, -4, -5, -9, -20, -59]:
    data = add_prices(data, n, '_%d_bday' % n, ffill_prices, use_index = False)
    data = add_prices(data, n, '_market_%d_bday' % n, ffill_prices, use_index = True)


# add unadjusted close price for scaling of fundamental values 
# (that are not corrected for dividends)
data = data.join(bfill_prices[['Price Close']], 
                 on = ['ticker_symbol', 'local_date'],
                 rsuffix = ' Unadj')
data['Price Close Unadj'] = data['Price Close']

data = data.join(bfill_prices[['Price Close']].groupby(level=0).shift(5), 
                 on = ['ticker_symbol', 'local_date'],
                 rsuffix = ' Unadj -5bd')
#data['Price Close Unadj'] = data['Price Close']



### compute FF-5 expected returns ###
print("Calculating FF-5 expected returns ...")

e_data = data.loc[data.mkt_index.isin(['SP500TR', 'SP600TR', 'SP400TR'])]#.sample(5000)

n_blocks = 4
e_data_split = np.array_split(e_data[['final_datetime', 'ticker_symbol']], n_blocks)
e_data_split = [(df, export_folder) for df in e_data_split]


with mp.Pool(n_blocks) as pool:
    p_er = pool.starmap(apply_fama_french_on_block, e_data_split)

exp_returns = pd.concat(p_er)

data = data.join(exp_returns)




### add dividend payment feature ###
print("Calculating dividends from prices ...")

prices = get_price_data(export_folder, list(data.ticker_symbol.unique()))

roll_dividend = prices.groupby('instrument_key')['div'].rolling('120d').sum()
roll_dividend = pd.DataFrame(roll_dividend)
roll_dividend.columns = ['TS Rolling Dividend']
roll_dividend.dropna(inplace = True)

roll_dividend.reset_index(inplace = True)
roll_dividend['local_date'] = roll_dividend.Date.dt.strftime('%Y-%m-%d')
roll_dividend.set_index(['instrument_key', 'local_date'], inplace = True)

data = data.join(roll_dividend['TS Rolling Dividend'], on = ['ticker_symbol', 'local_date'])
del prices, roll_dividend


### join market cap data ###

print("Joining market cap data ...")

mktcap = pd.read_hdf(export_folder + 'marketcap.hdf')

mktcap.sort_values('Date', inplace = True)
mktcap.set_index(['Date'], inplace = True)

mktcap = mktcap.groupby('instrument_key').resample('1d').ffill(5)
mktcap.reset_index(level = 1, inplace = True)

mktcap['local_date'] = mktcap.Date.dt.strftime('%Y-%m-%d')
mktcap.set_index(['instrument_key', 'local_date'], inplace = True)

data = data.join(mktcap[['Company Market Cap', 'Volume', 
                         'Outstanding Shares', 'Outstanding Shares Adj', 
                         'Price To Book Value Per Share']], 
                 on = ['ticker_symbol', 'local_date'])


rol_mktcap = mktcap[['Company Market Cap', 'Outstanding Shares', 'Outstanding Shares Adj']].sort_index()
rol_mktcap = rol_mktcap.groupby(level = 0).rolling(252, min_periods = 100)\
                .mean().reset_index(level = 0, drop = True)
rol_mktcap.columns = ['Rolling ' + c for c in rol_mktcap.columns]

data = data.join(rol_mktcap, on = ['ticker_symbol', 'local_date'])


del mktcap, rol_mktcap

gc.collect()


### join fundamental ratio data ###

print("Joining fundamental ratio data ...")

fundratio = pd.read_hdf(export_folder + 'fundratio.hdf')

fundratio.sort_values('Date', inplace = True)
fundratio.set_index(['Date'], inplace = True)

fundratio = fundratio.groupby('instrument_key').resample('1d').ffill(5)
fundratio.reset_index(level = 1, inplace = True)

fundratio['local_date'] = fundratio.Date.dt.strftime('%Y-%m-%d')
fundratio.set_index(['instrument_key', 'local_date'], inplace = True)

fundratio = fundratio[['Enterprise Value', 'Price To Tangible Book Value Per Share', 
                       'P/E', 'P/E/G', 'Enterprise Value To EBIT', 
                       'Enterprise Value To EBITDA', 'Price To Sales Per Share', 
                       'Enterprise Value To Sales', 'Price To Cash Flow Per Share',
                       'Enterprise Value To Operating Cash Flow',
                       'Total Debt To Enterprise Value', 'Net Debt To Enterprise Value',
                       'Dividend yield', 'Net Dividend Yield']]

data = data.join(fundratio, on = ['ticker_symbol', 'local_date'])
del fundratio




### compute FF size decile expected returns ###
print("Calculating FF size decile expected returns ...")

size_cuts = pd.read_csv(export_folder + '/F-F_ME_Breakpoints.CSV', 
                        skiprows = 1).dropna()
size_cuts['Date'] = pd.to_datetime(size_cuts['Month'], format = "%Y%m") \
                        + pd.tseries.offsets.MonthEnd(1)
size_cuts.set_index('Date', inplace = True)
size_cuts = size_cuts.resample('d').ffill()
size_cuts.set_index(size_cuts.index.strftime('%Y-%m-%d'), inplace = True)
size_cuts.drop(columns = ['Month', 'N'], inplace = True)
size_cuts['q00'] = 0.0
size_cuts['q100'] = 1E9

q_cols = ['q%02d' % q for q in range(0, 101, 5)]    

def get_quantile(x):
    if pd.isna(x['Company Market Cap']):
        return np.nan
    
    cuts = size_cuts.loc[x['local_date'], q_cols].tolist()
    
    return q_cols[pd.cut(x['Company Market Cap'] / 1E6, cuts, labels = False)]

data['FF ME Quantile'] = data.apply(get_quantile, axis = 1)


quantile_map = {
        np.nan: np.nan,
        'q00': 'Lo 10', 'q05': 'Lo 10', 
        'q10': 'Dec 2', 'q15': 'Dec 2',
        'q20': 'Dec 3', 'q25': 'Dec 3',
        'q30': 'Dec 4', 'q35': 'Dec 4',
        'q40': 'Dec 5', 'q45': 'Dec 5',
        'q50': 'Dec 6', 'q55': 'Dec 6',
        'q60': 'Dec 7', 'q65': 'Dec 7',
        'q70': 'Dec 8', 'q75': 'Dec 8',
        'q80': 'Dec 9', 'q85': 'Dec 9',
        'q90': 'Hi 10', 'q95': 'Hi 10', 'q100': 'Hi 10'
         }

data['FF Size Decile'] = data['FF ME Quantile'].apply(lambda q: quantile_map[q])

print("Done assigning deciles, computing returns ...")

e_data = data.loc[data.mkt_index.isin(['SP500TR', 'SP600TR', 'SP400TR'])]
e_data_split = np.array_split(e_data[['final_datetime', 'ticker_symbol', 
                                      'FF ME Quantile', 'FF Size Decile']], n_blocks)
e_data_split = [(df, export_folder) for df in e_data_split]

with mp.Pool(n_blocks) as pool:
    p_er = pool.starmap(apply_fama_french_decile_on_block, e_data_split)

dec_rets = pd.concat(p_er)

data = data.join(dec_rets)



### sanity checks ###

# find ticker symbolds missing in Eikon data
n = data.instrument_id.isna() & (data.ticker_symbol.isna() == False)
eikon_missing = data.loc[n, ['ticker_symbol', 'company_name']]
eikon_missing.drop_duplicates(subset = 'ticker_symbol', inplace = True)
eikon_missing.to_excel('eikon_missing.xlsx')

# find missing instruments from market index
index_missing = list(set(all_index_instruments) - set(data.instrument_id.values))
index_missing = pd.DataFrame(index_missing, columns = ['instrument_id'])

index_missing = index_missing.join(index.set_index('instrument_id'), 
                                   on = 'instrument_id')
index_missing.to_excel('index_missing.xlsx')


# check that report date is not after call date and not way before it
for c in ['EPS', 'REVENUE', 'BVPS', 'CFPS', 'DIVEST']:
    report_date = pd.to_datetime(data[c + ' Report Date'].dt.tz_convert('America/New_York').dt.date)
    invalid_report_date = (report_date > data.final_datetime) | \
                        ((data.final_datetime - report_date) > pd.Timedelta(7, 'd') )
    
    print("Dropping %d calls with %s report date after/more than 7 days before call" % \
              (invalid_report_date.sum(), c))
    data = data[~ invalid_report_date]



#data[~data.mkt_index.isna()].groupby([data.release_datetime.dt.year, 'mkt_index'])\
#                    ['EPS Actual'].apply(lambda x: pd.isna(x).sum()).unstack(1)

# check missing prices
#missing_prices = data[~data.mkt_index.isna() & data.Open.isna()]


# determine which S&P calls are missing in SA data
sa_data = data[~data.file_id.isna() & ~data.eikon_event_title.isna()]
sa_data = sa_data[['eikon_event_title']].drop_duplicates()
sa_data.set_index('eikon_event_title', inplace = True)
sa_data['call_available'] = True

data = data.join(sa_data, on = 'eikon_event_title')

sp_data = data[~data.mkt_index.isna()]

missing_sp_calls = sp_data[sp_data.file_id.isna() 
                            & (sp_data.eikon_f_year >= 2008)
                            & (sp_data.call_available != True)]

missing_sp_calls.drop_duplicates(subset='eikon_event_title', inplace = True)

missing_sp_calls[['ticker_symbol', 'local_date', 'eikon_event_title', 'eikon_f_year',
   'eikon_f_quarter', 'eikon_fiscal_period', 'fiscal_period',
   'mkt_index', 'rics', 'instrument_id', 'primary_ric']].to_excel('missing_sp_calls.xlsx')



data = add_sentiment_features(data)


print("Saving to temporary file %s" % tmp_file)
#data.to_hdf(tmp_file, 'data')
with open(tmp_file, "wb") as f:
    pickle.dump(data, f)

return tmp_file