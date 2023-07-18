
import zipline as zl
from zipline.finance import commission
import pytz
import datetime as dt
from constraints import DynamicTradeAtOpenOrCloseSlippageModel
import api
import logging

import zipline_ingest


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
chandler = logger.handlers[0]
cformatter = logging.Formatter('%(levelname)s - %(message)s')
chandler.setFormatter(cformatter)


capital_base = 100000

start = dt.datetime(2014, 3, 1, 0, 0, 0, 0, pytz.utc)
end = dt.datetime(2014, 3, 10, 0, 0, 0, 0, pytz.utc)




def initialize(context):
    """ here configuration of the backtester as well as setup of variables that 
    shall be available duringe execution """
    context.counter = 0 # a simple variable counting the days that have passed
    
    context.set_slippage(DynamicTradeAtOpenOrCloseSlippageModel(
            spread = 0.0, context = context))
    
    context.set_commission(commission.PerDollar(cost = 0.0005))



def handle_data(context, data):
    """ this method is called at the end of each trading day """
    """ here you have access to all historic data including that day, 
        your current portfolio and can place orders """
    
    current_datetime = zl.api.get_datetime()
    
    logging.debug("Current datetime: %s", current_datetime)
    
    pf_value = context.portfolio.portfolio_value
    logging.debug('Current portfolio value %f', pf_value)
    
    aapl = zl.api.symbol('AAPL')
    
    if context.counter == 3:
        logging.info('Last price of AAPL: %f', data.current(aapl, 'close'))
        logging.info("buying %s at %s", aapl, current_datetime)
        
        api.order(aapl, 10, context, order_on_open = True)
        api.order(aapl, -10, context, order_on_open = False)
    
    """ let's get some historic data """
    if context.counter == 10:
        historic_data = data.history(assets = aapl,
                                     fields = ['open', 'high', 'low', 'close', 'volume'], 
                                     bar_count = 50, 
                                     frequency = '1d')
        logging.info('Some history\n%s', historic_data[['close', 'volume']])
    
    
    """ log some info """
    zl.api.record(last_close_price_appl = data.current(aapl, 'close'))
    context.counter = context.counter + 1



result = zl.run_algorithm(start = start, end = end, initialize=initialize,
                          capital_base=capital_base, handle_data=handle_data,
                          bundle = 'eikon-data-bundle',
                          data_frequency='daily')


logging.info('Result - we better look at that in excel')    
result.index = result.index.strftime('%Y-%m-%d')
result['period_open'] = result['period_open'].dt.strftime('%Y-%m-%d')
result['period_close'] = result['period_close'].dt.strftime('%Y-%m-%d')
result.to_excel('zipline_example_result.xlsx')
