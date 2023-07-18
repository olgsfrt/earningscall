#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 09:12:56 2019

@author: mschnaubelt
"""

import urllib.request
import urllib.parse
import re
import pandas as pd
from bs4 import BeautifulSoup


SEARCH_SITE = 'https://www.reuters.com/finance/stocks/lookup?searchType=any&comSortBy=marketcap&sortBy=&dateRange=&search=%s'

seeking_alpha_list_file = '/mnt/data/earnings_calls/reuters_id.xlsx'

data = pd.read_excel(seeking_alpha_list_file)



def prepare_name(s):
    s = s.lower()
    s = s.replace("inc", "")
    s = s.replace(".", "")
    s = s.replace(",", "")
    s = s.replace("corp", "")
    
    s = re.sub('\(.*\)', '', s)
    
    return s.strip()


data['search_name'] = data.sa_name.apply(prepare_name)


#data = data.head(100)


def get_ticker(name):
    print("Getting for %s" % name)
    
    page = urllib.request.urlopen(SEARCH_SITE % urllib.parse.quote(name))
    bs = BeautifulSoup(page, 'html.parser')
    
    table = bs.find(lambda tag: tag.name=='table' 
                    and tag.has_attr('class')# and tag['class']=="search-table-data"
                    ) 
    
    if table is None:
        return pd.Series()
    
    rows = table.findAll(lambda tag: tag.name=='tr')
    
    results = [{
            'company': row.findAll(lambda tag: tag.name=='td')[0].contents[0],
            'ticker': row.findAll(lambda tag: tag.name=='td')[1].contents[0],
            'exchange': row.findAll(lambda tag: tag.name=='td')[2].contents[0]
            } 
        for row in rows if len(row.findAll(lambda tag: tag.name=='td')) == 3]
    
    results = pd.DataFrame(results)
    primary = results.head(1).iloc[0]
    
    return primary


r = data.search_name.apply(get_ticker)

data = data.join(r)
