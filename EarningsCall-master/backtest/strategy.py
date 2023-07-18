# -*- coding: utf-8 -*-

import logging
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay

from collections import defaultdict 

import zipline as zl


def call_strategy(bt, preds, min_rank, holding_days, 
                  allocation, long_short, per_call_fraction = 0.05):
    
    assert(allocation in ['per_call', 'per_day'])
    assert(long_short in ['long', 'short', 'long_short'])
    
    if 'reverse_list' not in dir(bt):
        bt.reverse_list = defaultdict(list)
    
    simu_local_date = bt.get_simulation_time().strftime('%Y-%m-%d')
    
    day_f = preds.local_date == simu_local_date
    rank_f = (preds.pred_rank <= min_rank) | (preds.pred_rank >= 1 - min_rank)
    
    f_preds = preds.loc[day_f & rank_f]
    
    if long_short == 'long':
        f_preds = f_preds[f_preds.pred_rank >= 0.5]
    if long_short == 'short':
        f_preds = f_preds[f_preds.pred_rank <= 0.5]
    
    logging.debug("\ttrading %d from %d calls", 
                  len(f_preds), len(preds.loc[day_f]))
    
    
    if len(f_preds) > 0:
        
        cal = zl.utils.calendars.get_calendar('NYSE')
        days_offset = CustomBusinessDay(holding_days - 1, 
                                        holidays=cal.adhoc_holidays, 
                                        calendar=cal.regular_holidays)
        
        reverse_date = bt.get_simulation_time() + days_offset
        reverse_date = reverse_date.strftime('%Y-%m-%d')
        
        if allocation == 'per_call':
            alloc_val = bt.initial_cash * per_call_fraction
        elif allocation == 'per_day':
            alloc_val = bt.initial_cash / holding_days / len(f_preds)
        
        for i, pred in f_preds.iterrows():
            signal = 2 * (pred.pred_rank >= 0.5) - 1
            
            try:
                call_ord = bt.order_value(pred.ticker_symbol, alloc_val*signal, 'O')
                bt.reverse_list[reverse_date].append(call_ord)
                
                mkt_ord = bt.order_value('SP1500ETF', - alloc_val*signal, 'O')
                bt.reverse_list[reverse_date].append(mkt_ord)
                
            except:
                logging.warn("Did not find symbol %s on date %s", 
                             pred.ticker_symbol, 
                             bt.get_simulation_time().strftime('%Y-%m-%d'))
    
    close_orders = bt.reverse_list[simu_local_date]
    for cls_ord in close_orders:
        order = zl.api.get_order(cls_ord)
        
        if order.created == bt.get_simulation_time():
            amount = order.amount
        else:
            amount = order.filled
            zl.api.cancel_order(order)
        
        bt.order_amount(order.sid.asset_name, - amount, 'C')



def week_strategy(bt, preds, k, holding_days, long_short = 'long_short'):
    assert(long_short in ['long', 'short', 'long_short'])
    
    if 'reverse_list' not in dir(bt):
        bt.reverse_list = defaultdict(list)
    
    simu_local_date = bt.get_simulation_time()
    
    if simu_local_date.weekday() == 4:
        week_dates = [(simu_local_date - n*pd.Timedelta('1d')).strftime('%Y-%m-%d') 
                        for n in [0, 1, 2, 3, 4]]
        
        day_f = preds.local_date.isin(week_dates)
        f_preds = preds.loc[day_f].sort_values('pred_p')
        
        logging.debug("\tweek has %d calls", len(f_preds))
        
        ak = len(f_preds) // 2 if len(f_preds) < 2*k else k
        n_top_calls = ak if 'long' in long_short else 0
        n_flop_calls = ak if 'short' in long_short else 0
        
        logging.debug("Trading %d short, %d long", n_flop_calls, n_top_calls)
        
        cal = zl.utils.calendars.get_calendar('NYSE')
        days_offset = CustomBusinessDay(holding_days - 1, 
                                        holidays=cal.adhoc_holidays, 
                                        calendar=cal.regular_holidays)
        
        reverse_date = bt.get_simulation_time() + days_offset
        reverse_date = reverse_date.strftime('%Y-%m-%d')
        
        logging.debug("Reversing at %s", reverse_date)
        
        alloc_val = bt.initial_cash / (2*k)
        
        for i, pred in f_preds.head(n_flop_calls).iterrows(): 
            order = bt.order_value(pred.ticker_symbol, - alloc_val, 'O')
            bt.reverse_list[reverse_date].append(order)
        
        for i, pred in f_preds.tail(n_top_calls).iterrows(): 
            order = bt.order_value(pred.ticker_symbol, alloc_val, 'O')
            bt.reverse_list[reverse_date].append(order)
    
    
    close_orders = bt.reverse_list[simu_local_date.strftime('%Y-%m-%d')]
    for cls_ord in close_orders:
        order = zl.api.get_order(cls_ord)
        
        if order.created == bt.get_simulation_time():
            amount = order.amount
        else:
            amount = order.filled
            zl.api.cancel_order(order)
        
        bt.order_amount(order.sid.asset_name, - amount, 'C')



def buy_and_hold_strategy(bt, preds):
    if 'bought' not in dir(bt):
        bt.order_value('SP1500ETF', bt.initial_cash, 'O')
    bt.bought = True


