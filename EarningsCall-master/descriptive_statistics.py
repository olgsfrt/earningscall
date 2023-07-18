#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:44:15 2019

@author: mschnaubelt
"""

import pandas as pd
import matplotlib.pyplot as plt

from util.prepare_data import prepare_data, clean_data, get_index_counts
from config import TARGETS, BASE_FEATURES, FEATURE_NAME_DICT


data = prepare_data()



writer = pd.ExcelWriter('descriptive_statistics.xlsx', engine = 'xlsxwriter')
tex_file = open('descriptive_statistics.tex', 'w')


pre_clean_counts, _ = get_index_counts(data)

data = clean_data(data)

post_clean_counts, _ = get_index_counts(data)

pre_clean_counts.to_excel(writer, sheet_name = 'pre_clean_counts')
post_clean_counts.to_excel(writer, sheet_name = 'post_clean_counts')

tex_file.writelines('\n### pre_clean_counts ###\n\n')
tex_file.writelines(pre_clean_counts.to_latex(float_format = lambda x: '%.1f' % x, 
                    multirow = False, escape = False))

tex_file.writelines('\n### post_clean_counts ###\n\n')
tex_file.writelines(post_clean_counts.to_latex(float_format = lambda x: '%.1f' % x, 
                    multirow = False, escape = False))


# comparison report date to call date
# report dates der versch. fundamentals vergleichen
# 


def mode(x):
    return x.mode()

def unique_count(x):
    return len(x.unique())

fiscal_period_dates = data.groupby('fiscal_period').local_date\
        .aggregate(['min', 'max', mode, unique_count])

fiscal_period_dates.to_excel(writer, sheet_name = 'local_date by fiscal_period')
tex_file.writelines('\n### local_date by fiscal_period ###\n\n')
tex_file.writelines(fiscal_period_dates.to_latex(float_format = lambda x: '%.4f' % x, 
                    multirow = False, escape = False))

cols = BASE_FEATURES + TARGETS
var_desc = data[cols].describe().transpose()
var_desc = pd.concat([var_desc, data[cols].quantile(0.05), data[cols].quantile(0.95)], axis = 1)
var_desc = var_desc[['count', 'mean', 'std', 0.05, '25%', '50%', '75%', 0.95, 'min', 'max']]

var_desc.to_excel(writer, sheet_name = 'variable statistics')
tex_file.writelines('\n### variable statistics ###\n\n')
tex_file.writelines(var_desc.to_latex(float_format = lambda x: '%.4f' % x, 
                    multirow = False, escape = False))

var_desc.index = var_desc.index.map(lambda v: FEATURE_NAME_DICT[v] if v in FEATURE_NAME_DICT.keys() else v)
var_desc = var_desc[['mean', 'std', 0.05, '25%', '50%', '75%', 0.95]]

tex_file.writelines('\n### variable statistics reduced ###\n\n')
tex_file.writelines(var_desc.to_latex(float_format = lambda x: '%.4f' % x, 
                    multirow = False, escape = False))


hours = data.release_datetime.dt.tz_convert('America/New_York').dt.hour
hour_dist = data.groupby(hours).local_date.count()

hour_dist.to_excel(writer, sheet_name = 'release hour distribution')
tex_file.writelines('\n### release hour distribution ###\n\n')
tex_file.writelines(hour_dist.to_latex(float_format = lambda x: '%.4f' % x, 
                    multirow = False, escape = False))


data['locale_date_hour'] = data.release_datetime.dt.tz_convert('America/New_York').dt.hour
data.loc[data.locale_date_hour <= 6,'locale_date_hour'] += 12


date_diffs = []
date_bins = [-10000, -366, -91, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 10, 100, 366, 10000]
date_bins = [pd.Timedelta(db, 'd') for db in date_bins]

for c in ['EPS', 'REVENUE', 'BVPS', 'CFPS', 'DIVEST']:
    dates = pd.to_datetime(data[c + ' Report Date'].dt.tz_convert('America/New_York').dt.date)
    
    diff = dates - data['final_datetime']
    diff.name = c
    
    dates = pd.concat([data[c + ' Report Date'], data[['final_datetime', 
                       'mkt_index', 'locale_date_hour']], diff], axis = 1)
    
    date_diff_hist = dates.groupby((pd.cut(diff, date_bins), 'mkt_index')).count()
    
    date_diffs.append(date_diff_hist)
    
    if c == 'EPS':
        eps_diff_data = dates

date_diffs = pd.concat(date_diffs, axis = 1).unstack(level = 1)


date_diffs.set_index(date_diffs.index.map(str)).to_excel(writer, sheet_name = 'report-call date distribution')

date_diffs = date_diffs.loc[:, [('EPS', 'SP500TR'), ('EPS', 'SP400TR'), ('EPS', 'SP600TR')]].dropna()
date_diffs.columns = date_diffs.columns.droplevel(0)
date_diffs.index = date_diffs.index.map(lambda c: -c.right.days)
date_diffs['Total'] = date_diffs.sum(axis = 1)

tex_file.writelines('\n### report vs. call date distribution ###\n\n')
tex_file.writelines(date_diffs.to_latex(float_format = lambda x: '%.0f' % x, 
                    multirow = False, escape = False))


eps_diff_data['days'] = eps_diff_data['EPS'].apply(lambda d: - d.days)


fig = plt.figure(figsize = (9, 3.5))
ax = fig.subplots(1, 1)

bar_data = eps_diff_data.groupby(('days', 'locale_date_hour')).locale_date_hour.count()

last_values = pd.Series(0, index = bar_data.index.levels[1])
for d_min, d_max, d_label in [(0, 0, 'same day'), (1, 1, '1 day'), (2, 10, '>1 day')]:
    day_data = bar_data.loc[[slice(d_min, d_max)], :].sum(level = 1, axis = 0)
    
    bp = ax.bar(day_data.index.values, day_data, 0.8, 
                bottom = last_values[day_data.index], 
                label = d_label)
    last_values += day_data

ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
ax.set_xlim(-0.5, 23.5)

ax.set_ylabel('Number of earnings calls', size = 13)
ax.set_xlabel('Hour of the day (Eastern Time)', size = 13)

#ax.ticklabel_format(style='sci', axis='y', scilimits=(0,3))
#ax.yaxis.major.formatter._useMathText = True

ax.tick_params(axis = 'both', which = 'major', labelsize = 12)

ax.legend(fontsize = 12)

fig.tight_layout()
fig.subplots_adjust(hspace = 0.0, wspace = 0.0,
                    left = 0.09, right = 0.99,
                    bottom = 0.14, top = 0.99)

fig.savefig('hour_distribution.pdf')
plt.close()


data[cols].corr(method = 'pearson').to_excel(writer, sheet_name = 'correlations Pearson')
data[cols].corr(method = 'spearman').to_excel(writer, sheet_name = 'correlations Spearman')


writer.save()
tex_file.close()

