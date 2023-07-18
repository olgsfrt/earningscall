#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:04:36 2019

@author: mschnaubelt
"""

import pandas as pd
import inspect, os

import logging

import zipline as zl
import pyfolio as pf

from backtest import api as art_api
from backtest.constraints import DynamicTradeAtOpenOrCloseSlippageModel
from backtest.constraints import PerDollarMarketSpecific

from backtest.strategy import call_strategy, buy_and_hold_strategy, week_strategy


class Backtester:
    
    def __init__(self, backtest_config):
        self._config = backtest_config
        
    
    """
    Run the backtest 
    """
    def run(self, initial_cash: float, predictions: pd.DataFrame, **kwargs) -> pd.DataFrame:
        assert(initial_cash >= 0)
        assert(isinstance(self._config['start'], pd.Timestamp))
        assert(self._config['start'].tzinfo != None)
        assert(isinstance(self._config['end'], pd.Timestamp))
        assert(self._config['end'].tzinfo != None)
        
        if 'bundle' in kwargs:
            bundle = kwargs['bundle']
        else:
            bundle = 'eikon-data-bundle'
        
        path = os.path.dirname(os.path.abspath(
                inspect.getfile(inspect.currentframe())))
        
        self._predictions = predictions
        
        self.initial_cash = initial_cash
        
        self.result = zl.run_algorithm(start = self._config['start'], 
                                       end = self._config['end'],
                                       initialize = self._initialize,
                                       capital_base = initial_cash,
                                       handle_data = self._handle_data,
                                       bundle = bundle,
                                       extensions = [path + '/zipline_ingest.py'],
                                       data_frequency = 'daily')
        
        self.result.index = self.result.index.tz_localize(None).normalize()
        
    
    
    def _initialize(self, context):
        self.current_context = context
        
        self._strategy = self._config['strategy']
        
        zl.api.set_cancel_policy(zl.api.NeverCancel())
        
        context.set_slippage(DynamicTradeAtOpenOrCloseSlippageModel(
            spread = 0.0, context = context))
        
        context.set_commission(PerDollarMarketSpecific(
                market_asset = zl.api.symbol('SP1500ETF'),
                default_cost = self._config['default_commission'],
                cost_dict = self._config['commission_dict'],
                market_cost = self._config['market_commission']))
    
    
    def _handle_data(self, context, data):
        self.current_context = context
        self.current_data = data
        
        self._current_time = zl.api.get_datetime()
        logging.debug("running at %s", self._current_time.tz_convert('America/New_York'))
        
        self._strategy(self, self._predictions, **self._config['strategy_args'])
    
    
    def get_context(self):
        return self.current_context
    
    def get_simulation_time(self) -> pd.Timestamp:
        return self._current_time
    
    def get_results(self):
        return self.result.copy()
    
    def order_value(self, instrument_key, value, on):
        zl_symbol = zl.api.symbol(instrument_key)
        
        return art_api.order_value(zl_symbol, value, self.current_context, 
                      order_on_open = (on == 'O'))
    
    def order_amount(self, instrument_key, amount, on, stop_price = None):
        zl_symbol = zl.api.symbol(instrument_key)
        
        return art_api.order(zl_symbol, amount, self.current_context, 
                      order_on_open = (on == 'O'), 
                      stop_price = stop_price)
    
    def get_transactions(self):
        result = []
        for daily_list in self.result.transactions:
            for t in daily_list:
                result.append((t['order_id'],
                            t['dt'].tz_convert('America/New_York'),
                            t['sid'].symbol,
                            t['amount'],
                            t['price'],
                            t['amount']*t['price']))
        result = pd.DataFrame(result, columns = ['order_id', 'dt', 'sid', 
                                                 'amount', 'price', 'volume'])
        
        return result
    
    def get_performance_stats(self):
        result = self.get_results()
        return pf.timeseries.perf_stats(result.returns)
    



if __name__ == "__main__":
    backtest_config = {
        'default_commission': 0.001,
        'market_segment_commission': {'SP500TR': 0.001, 'SP400TR': 0.0015, 'SP600TR': 0.002},
        'market_commission': 0.0001,
        'start': pd.Timestamp('2013-01-01', tz = 'America/New_York'),
        'end': pd.Timestamp('2013-12-31', tz = 'America/New_York'),
        'strategy': call_strategy,
        'strategy_args': {
                'min_rank': 0.1,
                'holding_days': 5,
                'allocation': 'per_day',
                'per_call_fraction': 0.02,
                'long_short': 'long_short'
                }
        }
    
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    chandler = logger.handlers[0]
    cformatter = logging.Formatter('%(levelname)s - %(message)s')
    chandler.setFormatter(cformatter)
    
    predictions = pd.read_hdf('/mnt/data/earnings_calls/runs/run-2019-09-04T10_07_40/job-0/predictions.hdf')
    #predictions = predictions[predictions.local_date <= '2013-06-10']
    
    
    share_costs = predictions.groupby('ticker_symbol').mkt_index.max().apply(lambda x: 
        backtest_config['market_segment_commission'][x]).to_dict()
    
    backtest_config['commission_dict'] = share_costs
    
    
    backtester = Backtester(backtest_config)
    backtester.run(1E9, predictions)
    
    result = backtester.get_results()
    stats = backtester.get_performance_stats()
    trans = backtester.get_transactions()

