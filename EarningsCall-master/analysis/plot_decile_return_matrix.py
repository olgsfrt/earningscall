#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:57:40 2019

@author: mschnaubelt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:52:52 2019

@author: mschnaubelt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import FINAL_RUN_FOLDERS
from analysis_helper import read_run


MODEL_NAMES = {'0 ES-EP-surprise': 'ES', '8 RF-D20-E5000': 'RF', '1 LR': 'LR', '1 LR-B10': 'LR-B'}

SELECTED_JOBS1 = [{
        'index': (m, hp, 'FE+POL+UIQ+VR+54'),
        'period': hp,
        'model': MODEL_NAMES[m],
        'row': r,
        'column': c
        } for r, hp in enumerate([5, 20, 60]) 
            for c, m in enumerate(['0 ES-EP-surprise', '1 LR-B10', '8 RF-D20-E5000'])]

SELECTED_JOBS2 = [{
        'index': (m, hp, 'FE+POL+UIQ+VR+54'),
        'period': hp,
        'model': MODEL_NAMES[m],
        'row': r,
        'column': c
        } for r, hp in enumerate([5, 60]) 
            for c, m in enumerate(['0 ES-EP-surprise', '1 LR-B10', '8 RF-D20-E5000'])]

SELECTED_JOBS3 = [{
        'index': ('8 RF-D20-E5000', hp, 'FE+POL+UIQ+VR+54'),
        'period': hp,
        'model': '',
        'row': 0,
        'column': c
        } for c, hp in enumerate([5, 60])]

def plot_deciles(job_results, ax):
    dr = job_results.deciles_returns
    
    ax.axhline(0, color = 'lightgray')
    
    ax.bar([str(i) for i in dr.index.values], dr['mean']*10000, )
           #color = 'blue')
    
    ax.yaxis.tick_right()
    
    y_min, y_max = 0, 0
    for i, p in enumerate(ax.patches):
        text = "%.2f\n(%.1f)" % (dr.iloc[i]['mean']*10000, dr.iloc[i].t)
        y = dr.iloc[i]['mean']*10000
        ax.annotate(text, 
                    (p.get_x() + p.get_width()/2, y + 5 if y > 0 else y - 5), 
                    fontsize = 10, ha = "center", va = 'bottom' if y > 0 else 'top')
        y_min, y_max = min(y_min, y), max(y_max, y)
    
    ax.set_ylim(dr['mean'].min()*10000 - 10, dr['mean'].max()*10000 + 10)
    
    ax.tick_params(axis = "y", direction = "in")
    ax.tick_params(axis = "x", direction = "in")
    
    return y_min, y_max
    


def process_run(run_folder, jobs_config, filename = 'decile_matrix.pdf'):
    results = read_run(run_folder)
    
    rows = max([c['row'] for c in jobs_config]) + 1
    cols = max([c['column'] for c in jobs_config]) + 1
    
    
    fig = plt.figure(figsize = (12, 8/2.5 * rows))
    
    axs = fig.subplots(rows, cols, sharex = True, sharey = True)
    if len(np.shape(axs)) == 1:
        axs = np.array([axs])
    
    global_y_min, global_y_max = 0, 0
    
    for job in jobs_config:
        job_results = results.loc[job['index']]
        ax = axs[job['row'], job['column']]
        
        y_min, y_max = plot_deciles(job_results, ax)
        
        global_y_min, global_y_max = min(global_y_min, y_min), max(global_y_max, y_max)
    
    #global_y_min = - global_y_max
    
    for job in jobs_config:
        ax = axs[job['row'], job['column']]
        ax.set_ylim(global_y_min - 65, global_y_max + 65)
        
        if job['column'] == cols - 1:
            ax.yaxis.set_label_position("right")
            #ax.set_ylabel('Mean abnormal return', fontsize = 12)
        
        if job['row'] == rows - 1:
            ax.set_xlabel('decile', fontsize = 12)
            ax.set_xticklabels(range(1, 11))
        
        if job['row'] == 0:
            ax.annotate('%s' % job['model'],
                        xy = (0.5, 1.0), xycoords = 'axes fraction',
                        size = 13, ha = 'center', va = 'bottom')
        
        #if job['column'] == 0:
        ax.annotate('$T=%s$' % job['period'],
                        xy = (0.05, 0.9), xycoords = 'axes fraction',
                        size = 13, ha = 'left', va = 'center')
        
        ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
        
        ax.set_yticks([])
    #plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")
    
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.0, wspace = 0.0,
                        left = 0.025, right = 0.98,
                        bottom = 0.08, top = 0.95)
    
    fig.savefig(run_folder + filename)
    plt.close()
    
    


if __name__ == '__main__':
    for run_folder in FINAL_RUN_FOLDERS:
        process_run(run_folder, SELECTED_JOBS1)
        process_run(run_folder, SELECTED_JOBS2, 'decile_matrix_small.pdf')
        process_run(run_folder, SELECTED_JOBS3, 'decile_matrix_rf.pdf')



