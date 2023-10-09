#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:12:32 2017

@author: thomas
"""


import zipline as zl
from backtest.constraints import DynamicTradeAtOpenOrCloseSlippageModel


def order(asset, amount, context, order_on_open, limit_price=None, stop_price=None, style=None):
    #if order_on_open:
        #assert(isinstance(context.blotter.slippage_models[zl.assets._assets.Equity], 
        #                  DynamicTradeAtOpenOrCloseSlippageModel))
    
    order_id = zl.api.order(asset, amount, limit_price, stop_price, style)
    if order_on_open:
        context.on_open_orders.append(order_id)
    return order_id
    
def order_percent(asset, percent, context, order_on_open, limit_price=None, stop_price=None, style=None):
    if order_on_open:
        assert(isinstance(context.blotter.slippage_models[zl.assets._assets.Equity], 
                          DynamicTradeAtOpenOrCloseSlippageModel))


    order_id = zl.api.order_percent(asset, percent, limit_price, stop_price, style)
    if order_on_open:
        context.on_open_orders.append(order_id)
    return order_id
 

def order_value(asset, value, context, order_on_open, limit_price=None, stop_price=None, style=None):
    #if order_on_open:
    #    assert(isinstance(context.blotter.slippage_models[zl.assets._assets.Equity], 
    #                      DynamicTradeAtOpenOrCloseSlippageModel))


    order_id = zl.api.order_value(asset, value, limit_price, stop_price, style)
    if order_on_open:
        context.on_open_orders.append(order_id)
    return order_id
    


def order_target(asset, target, context, order_on_open, limit_price=None, stop_price=None, style=None):
    if order_on_open:
        assert(isinstance(context.blotter.slippage_models[zl.assets._assets.Equity], 
                          DynamicTradeAtOpenOrCloseSlippageModel))


    order_id = zl.api.order_target(asset, target, limit_price, stop_price, style)
    if order_on_open:
        context.on_open_orders.append(order_id)
    return order_id



def order_target_value(asset, target, context, order_on_open, limit_price=None, stop_price=None, style=None):
    if order_on_open:
        assert(isinstance(context.blotter.slippage_models[zl.assets._assets.Equity], 
                          DynamicTradeAtOpenOrCloseSlippageModel))


    order_id = zl.api.order_target_value(asset, target, limit_price, stop_price, style)
    if order_on_open:
        context.on_open_orders.append(order_id)
    return order_id    
    


def order_target_percent(asset, target, context, order_on_open, limit_price=None, stop_price=None, style=None):
    if order_on_open:
        assert(isinstance(context.blotter.slippage_models[zl.assets._assets.Equity], 
                          DynamicTradeAtOpenOrCloseSlippageModel))


    order_id = zl.api.order_target_percent(asset, target, limit_price, stop_price, style)
    if order_on_open:
        context.on_open_orders.append(order_id)
    return order_id    
    
    