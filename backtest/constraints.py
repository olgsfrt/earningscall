#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:57:45 2017

@author: thomas
"""

from zipline.finance import slippage
from zipline.finance.commission import EquityCommissionModel


class DynamicTradeAtOpenOrCloseSlippageModel(slippage.SlippageModel):
    def __init__(self, spread, context):
        self.spread = spread # Store bid/ask spread
        context.on_open_orders = []
        self.on_open_orders = context.on_open_orders

    def process_order(self, data, order):
        if order.id in self.on_open_orders:
#                print('executing on open order')
            targetExecutionPrice = data.current(order.sid, 'open')
        else:
#                print('executing on close order')
            targetExecutionPrice = data.current(order.sid, 'close')
        # Apply spread slippage
        targetExecutionPrice += self.spread * order.direction
        # Create the transaction using the new price we've calculated.
        return (targetExecutionPrice, order.amount)


class PerDollarMarketSpecific(EquityCommissionModel):
    """
    Calculates a commission for a transaction based on a per dollar cost.

    Parameters
    ----------
    cost : float
        The flat amount of commissions paid per dollar of equities traded.
    market_cost : float
        The flat amount of commissions paid per dollar market hedge.
    """
    def __init__(self, market_asset, default_cost, cost_dict, market_cost):
        self.market_asset = market_asset
        self.default_cost_per_dollar = float(default_cost)
        self.cost_dictionary = cost_dict
        self.market_cost_per_dollar = float(market_cost)
    
    def __repr__(self):
        return ("{class_name}(default_cost_per_dollar={cost}," + 
                " cost_per_dollar_market_hedge={market_cost})").format(
            class_name=self.__class__.__name__,
            cost=self.default_cost_per_dollar,
            market_cost=self.market_cost_per_dollar)
    
    def calculate(self, order, transaction):
        """
        Pay commission based on dollar value of shares.
        """
        
        if order.asset == self.market_asset:
            cost_per_share = transaction.price * self.market_cost_per_dollar
        else:
            cost = self.default_cost_per_dollar
            if order.asset.symbol in self.cost_dictionary:
                cost = self.cost_dictionary[order.asset.symbol]
            
            cost_per_share = transaction.price * cost
        
        return abs(transaction.amount) * cost_per_share
