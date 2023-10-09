# -*- coding: utf-8 -*-


from logbook import Logger

import numpy as np
import pandas as pd

from zipline.data.bundles import register
from zipline.utils.calendars import get_calendar


EXPORT_FOLDER = '/mnt/data/earnings_calls/export/'

log = Logger('zipline_ingest.py')


def bundle_hf_data(price_file, debug = False):
    
    def ingest(environ,
               asset_db_writer,
               minute_bar_writer,
               daily_bar_writer,
               adjustment_writer,
               calendar,
               cache,
               show_progress,
               output_dir,
               start,
               end):
        
        log.info("Starting bundle build from %s" % price_file)
        
        data = pd.read_hdf(price_file)
        data.dropna(subset = ['Open', 'Close'], inplace = True)
        
        data = data.loc[data.Currency == 'USD']
        data['instrument_key'] = data.instrument_key.str.upper()
        
        log.info("Importing %d instruments" % len(data.instrument_key.unique()))
        
        dfMetadata = []
        
        def read_instruments():
            for sid, (instrument_key, instrument_data) in enumerate(data.groupby('instrument_key')):
                log.debug("Reading instrument %s" % instrument_key)
                
                log.debug("\tInstrument has %d rows" % len(instrument_data))
                
                if len(instrument_data) == 0:
                    log.debug("\tNo data for instrument, skipping")
                    continue
                
                instrument_data.drop_duplicates(subset = ['Date'], inplace = True)
                
                instrument_data.set_index('Date', inplace = True)
                instrument_data.sort_index(inplace = True)
                
                #dfData['exchange_open'] = instrument_data.index.map(calendar.is_open_on_minute)
                #dfData = dfData[dfData['exchange_open'] == True]
                
                start_date = instrument_data.index[0]
                log.debug("\tstart_date %s" % start_date)
                
                end_date = instrument_data.index[-1]
                log.debug("\tend_date %s" % end_date)
                
                ac_date = end_date + pd.Timedelta(days=1)
                log.debug("\tac_date %s" % ac_date)
                
                sessions = get_calendar('NYSE').sessions_in_range(start_date, end_date)
                instrument_data = instrument_data.reindex(sessions)
                
                # Update our meta data
                dfMetadata.append((sid, instrument_key, start_date, end_date, \
                                        ac_date, instrument_key, "Eikon"))
                
                instrument_data['High'] = np.nan
                instrument_data['Low'] = np.nan
                instrument_data['Volume'].fillna(1.0, inplace = True)
                
                instrument_data = instrument_data.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']]
                instrument_data.columns = ['open', 'high', 'low', 'close', 'volume']
                instrument_data = instrument_data.astype(float)
                
                yield (sid, instrument_data)
                
                if debug:
                    break
        
        
        liData = read_instruments()
        
        
        log.info("calling daily_bar_writer")
        daily_bar_writer.write(liData, 
                                show_progress = True)
        log.info("returned from daily_bar_writer")
        
        dfMetadata = pd.DataFrame(dfMetadata, 
                                  columns=['sid', 'asset_name', 'start_date', 
                                           'end_date', 'auto_close_date', 
                                           'symbol', 'exchange'])\
                        .set_index('sid')
        
        log.info("calling asset_db_writer")
        log.info(dfMetadata)
        asset_db_writer.write(equities = dfMetadata)
        log.info("returned from asset_db_writer")
        
        log.info("calling adjustment_writer")
        adjustment_writer.write()
        log.info("returned from adjustment_writer")
        
    return ingest


register(
    'eikon-data-bundle',
    bundle_hf_data(price_file = EXPORT_FOLDER + "/adjusted_prices.hdf", 
                   debug = False),
)
