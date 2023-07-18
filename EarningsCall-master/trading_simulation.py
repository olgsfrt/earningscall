#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:01:05 2019

@author: mschnaubelt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 10:56:38 2018

@author: mschnaubelt
"""

import logging
import time
import empyrical
import pandas as pd
import os
#import matplotlib
#matplotlib.use('PDF')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from backtest.backtester import Backtester


def run_backtest(predictions, backtest_config, out_path):
    logging.info("Running backtest %s", 
                 backtest_config['name']if 'name' in backtest_config else backtest_config)
    
    t0 = time.time()
    
    
    if 'backtest_subset' in backtest_config and \
            backtest_config['backtest_subset'] is not None:
        
        index_names = ['SP500TR', 'SP400TR', 'SP600TR'] \
                    if backtest_config['backtest_subset'] == 'SP1500' \
                    else [backtest_config['backtest_subset']]
        predictions = predictions.loc[predictions.mkt_index.isin(index_names)]
    
    share_costs = predictions.groupby('ticker_symbol').mkt_index.max().apply(lambda x: 
    backtest_config['market_segment_commission'][x]).to_dict()
    
    backtest_config['commission_dict'] = share_costs
    
    backtester = Backtester(backtest_config)
    backtester.run(1E9, predictions)
    
    result = backtester.get_results()
    stats = backtester.get_performance_stats()
    trans = backtester.get_transactions()
    
    #result['daily_r'] = (result['pnl'] / result['cumulated_used_cash']).fillna(0)
    #result['cash_loadfactor'] = result['cumulated_used_cash'] / result['daily_start_cash']
    
    export_result = result.copy()
    for c in ['period_open', 'period_close']:
        export_result[c] = export_result[c].apply(str)
    export_result.to_excel(out_path + '/results.xlsx')
    
    result.drop(['orders', 'transactions'], axis = 1, inplace = True)
    result.to_hdf(out_path + '/results.hdf', 'results', complevel = 9, complib = 'bzip2')
    
    trans['dt'] = trans['dt'].apply(str)
    trans.to_excel(out_path + '/transactions.xlsx')
    
    fig = plt.figure(figsize = (25, 8))
    ax1 = fig.subplots(1, 1)
    
    ax1.plot(result.returns.cumsum() + 1)
    ax1.grid(True)
    
    title = os.path.split(out_path)[1]
    fig.suptitle(title)
    
    fig.tight_layout(rect = [0, 0.03, 1, 0.95])
    fig.savefig(out_path + '/cumsum.pdf')
    plt.close(fig)
    
    
    dt = time.time() - t0
    logging.info("Runtime: %d seconds", dt)
    
    
    reg = smf.ols('returns ~ 1', data = result).fit(
            cov_type = 'HAC', cov_kwds = {'maxlags': 10})
    
    result_dict = {
                   'PnL': 
                result['pnl'].sum(),
                    'Mean daily return':
                result['returns'].mean(),
                    'Std daily return':
                result['returns'].std(),
                    'Mean daily standard error (NW)': 
                reg.HC0_se[0],
                    'Mean daily t-statistic (NW)':
                reg.params[0] / reg.HC0_se[0],
                    'Median daily return':
                result['returns'].median(),
                    '25% daily return': 
                result['returns'].quantile(0.25),
                    '75% daily return': 
                result['returns'].quantile(0.75),
                    'Max daily return':
                result['returns'].max(),
                    'Min daily return':
                result['returns'].min(),
                    'Skewness':
                result['returns'].skew(),
                    'Kurtosis':
                result['returns'].kurtosis(),
                    'Max leverage':
                result.max_leverage.max(),
                    'hist. VaR 1%': 
                empyrical.value_at_risk(result['returns'], 0.01),
                    'hist. CVaR 1%':
                empyrical.conditional_value_at_risk(result['returns'], 0.01),
                    'hist. VaR 5%': 
                empyrical.value_at_risk(result['returns'], 0.05),
                    'hist. CVaR 5%': 
                empyrical.conditional_value_at_risk(result['returns'], 0.05),
                    'share with return >= 0': 
                (result['returns'] >= 0.0).mean(),
                    'runtime': 
                dt,
                #    'Mean cash loadfactor': 
                #result.cash_loadfactor.mean(),
                #    'days with trades': 
                #(result.cash_loadfactor > 0).sum()
                }
    
    for y in [2014, 2015, 2016, 2017, 2018]: 
        f = result.index.year == y
        result_dict.update({
                'Annual return %d' % y:
                    empyrical.annual_return(result[f]['returns']),
                'Sharpe ratio %d' % y:
                    empyrical.sharpe_ratio(result[f]['returns'])
            })
    
    result_dict.update({k: str(v) for k, v in backtest_config.items()})
    result_dict.update(stats.to_dict())
    
    
    pd.Series(result_dict).to_hdf(out_path + '/stats.hdf', 'stats')
    
    return result_dict, result


def save_backtest_results(context, results):
    result_df = pd.DataFrame(results)
    cols = ['backend','name','stock_commission','market_commission',
            'bin_period','trade_period','prediction_file',
            'args','runtime','PnL',
            'Sharpe ratio OCU', 'Annual return OCU',
            'Mean cash loadfactor', 'days with trades',
            'Annual return','Annual volatility','Sharpe ratio','Calmar ratio',
            'Mean daily return','Median daily return',
            'Max daily return','Min daily return',
            'Annual return 2014', 'Sharpe ratio 2014',
            'Annual return 2015', 'Sharpe ratio 2015',
            'Cumulative returns','Daily value at risk','Kurtosis','Max drawdown',
            'Skew','Sortino ratio','Stability','Tail ratio',
            'Mean trade return','Trade roundtrips','Mean SPY return',
            'Mean long SPY return', 'Mean short SPY return',
            'Std SPY return','Mean long trade return','Trade long roundtrips',
            'Mean short trade return','Trade short roundtrips']
    result_df = result_df[cols]
    
    result_df.to_excel(context.outputdir + "results.xlsx")
    
    del result_df

