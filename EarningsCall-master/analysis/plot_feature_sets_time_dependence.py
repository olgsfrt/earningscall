#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:59:53 2019

@author: mschnaubelt
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import FINAL_TIME_DEP_RUN, FEATURE_NAME_DICT
from config import FEATURE_SETS_ORDER, FEATURE_SETS_DICT
from analysis.analysis_helper import read_run, extract_job_info
from analysis.feature_importance import analyze_feature_importance


RUN_FOLDER = FINAL_TIME_DEP_RUN


DEC_MEAN_FN = lambda x, d: x['deciles_returns'].iloc[d]['mean'] * 100
DEC_MEANERR_FN = lambda x, d: x['deciles_returns'].iloc[d]['mean'] / x['deciles_returns'].iloc[d]['t'] * 100

MEAN_FN = lambda x: x['mean'] * 100
MEANERR_FN = lambda x: x['mean'] / x['t'] * 100

TF_MEAN_FN = lambda x: x['top_flop_mean'] * 100
TF_MEANERR_FN = lambda x: x['top_flop_mean'] / x['top_flop_t'] * 100

TF_ACC = lambda x: x['top_flop_bacc']*100
ACC = lambda x:  x['SP1500_results']['test_bacc']*100


COLORS = {'VR': 'C1', 'FE': 'C0', 'UIQ': 'C2', 'POL': 'C3', 'all': 'k'}
STYLES = {'all': 'solid', 'VR': (0, (3, 6)), 'FE': (0, (3, 3)),
          'UIQ': (0, (2, 3, 1, 3)), 'POL': (0, (2, 3, 1, 3, 1, 3))}

SEL_TIMES = [-5, -4, -3, -2, -1, 
             1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60]


def plot_series(ax, results, flip_neg_time = True, 
                model = '8 RF-D20-E5000', feature_set = 'FE', 
                value_fn = MEAN_FN, error_fn = None, 
                error_band = 0.90, color = None,
                linestyle = None,
                sel_times = SEL_TIMES,
                label = ''):
    
    I = (results.index.get_level_values('features') == feature_set) & \
            (results.index.get_level_values('model') == model)
    R = results.loc[I].sort_index()
    
    if sel_times is not None:
        R = R.loc[(slice(None), sel_times, slice(None)), :]
    
    P = R.index.get_level_values('period')
    V = R.apply(value_fn, axis = 1).astype(float)
    
    if flip_neg_time:
        is_surprise_model = R['job'].apply(lambda x: 'EarningsSurprise' in str(x['model']))
        V *= np.where(~ is_surprise_model & (P < 0), -1, 1)
    
    ax.plot(P, V, label = label, color = color, linestyle = linestyle)
    
    if error_fn is not None:
        k = {0.99: 2.626, 0.9: 1.66}[error_band]
        STD = R.apply(error_fn, axis = 1).astype(float)
        ax.fill_between(P, V - k*STD, V + k*STD, alpha = 0.35)


def plot_decile_returns_acc(results, run_folder):
    SP = [('FE', 'FE'), ('VR', 'VR'), ('FE+POL+UIQ+VR+54', 'all')]
    SP = [('FE+POL+UIQ+VR+54', 'all')]
    
    fig = plt.figure(figsize = (9, 6))
    axs = fig.subplots(len(SP) + 1, 1, sharex = True, 
                       gridspec_kw = {'height_ratios': [2, 1]})
    
    for ax in axs:
        ax.axvspan(-10, 0.5, alpha = 0.5, color = 'lightgray')
        
        ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
        ax.set_xlim(-5, 60)
        
        ax.axhline(0.0, color = 'gray')
        ax.set_ylim(-5.8, 5.8)
    
    
    for i, (fs, fl) in enumerate(SP): 
        for d in [9, 8, 1, 0]:
            kwargs = { 'model': '8 RF-D20-E5000', 'value_fn': lambda x: DEC_MEAN_FN(x, d), 
                      'error_fn': lambda x: DEC_MEANERR_FN(x, d) }
            plot_series(axs[i], results, feature_set = fs, 
                        label = 'decile %d' % (d + 1), **kwargs)
            axs[i].set_ylabel('CAR', fontsize = 13)
    
    
    kwargs = { 'model': '8 RF-D20-E5000', 'value_fn': TF_ACC, 'flip_neg_time': False }
    plot_series(axs[-1], results, feature_set = 'FE+POL+UIQ+VR+54', 
                label = 'deciles 1 and 10', **kwargs)
    
    kwargs = { 'model': '8 RF-D20-E5000', 'value_fn': ACC, 'flip_neg_time': False }
    plot_series(axs[-1], results, feature_set = 'FE+POL+UIQ+VR+54', 
                label = 'all deciles', **kwargs)
    
    axs[-1].set_ylim(0.5, 0.85)
    axs[-1].set_ylabel('accuracy', fontsize = 13)
    
    axs[-1].set_xlabel('event time', fontsize = 13)
    axs[0].legend(loc = 'upper center', prop = {'size': 13}, ncol = 4, frameon = False)
    axs[-1].legend(loc = 'upper center', prop = {'size': 13}, ncol = 4, frameon = False)
    
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.0, wspace = 0.0,
                        left = 0.08, right = 0.98,
                        bottom = 0.08, top = 0.98)
    
    fig.savefig(run_folder + 'time_dep_dec_acc.pdf')
    plt.close()


def plot_mean_return(results, run_folder):
    fig = plt.figure(figsize = (10, 4.5))
    axs = fig.subplots(2, 1, sharex = True, gridspec_kw = {'height_ratios': [2, 1.5]})
    
    for ax in axs:
        ax.axvspan(-10, 0, alpha = 0.5, color = 'lightgray')
        
        ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
        ax.set_xlim(1, 60)
    
    #axs[0].axhline(0.0, color = 'gray')
    
    axs[0].set_ylim(-0.1, 2.4)
    
    kwargs = { 'model': '8 RF-D20-E5000', 'value_fn': MEAN_FN, 'error_fn': MEANERR_FN }
    plot_series(axs[0], results, feature_set = 'FE+POL+UIQ+VR+54', 
                label = 'all events', **kwargs)
    
    kwargs = { 'model': '8 RF-D20-E5000', 'value_fn': TF_MEAN_FN, 'error_fn': TF_MEANERR_FN }
    plot_series(axs[0], results, feature_set = 'FE+POL+UIQ+VR+54', 
                label = 'top-flop events', linestyle = 'dashed', **kwargs)
    axs[0].set_ylabel('Mean abnormal return', fontsize = 13)
    
    
    kwargs = { 'model': '8 RF-D20-E5000', 'value_fn': ACC, 'flip_neg_time': False }
    plot_series(axs[1], results, feature_set = 'FE+POL+UIQ+VR+54', 
                label = 'top-flop events', **kwargs)
    
    kwargs = { 'model': '8 RF-D20-E5000', 'value_fn': TF_ACC, 'flip_neg_time': False }
    plot_series(axs[1], results, feature_set = 'FE+POL+UIQ+VR+54', 
                label = 'all events', linestyle = 'dashed', **kwargs)
    
    
    axs[1].set_ylabel('Dir. accuracy', fontsize = 13)
    axs[1].set_ylim(50, 57.9)
    axs[1].set_yticks([51, 54, 57])
    
    axs[1].set_xlabel('Forecast horizon $T$', fontsize = 13)
    axs[1].set_xticks([1, 10, 20, 30, 40, 50 , 60])
    
    axs[0].legend(loc = 'upper left', prop = {'size': 12}, ncol = 4, frameon = False)
    
    fig.align_labels()
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.0, wspace = 0.02,
                        left = 0.07, right = 0.98,
                        bottom = 0.11, top = 0.98)
    
    fig.savefig(run_folder + 'time_dep_mean_acc.pdf')
    plt.close()



def plot_mean_return_fs(results, run_folder):
    fig = plt.figure(figsize = (10, 5))
    axs = fig.subplots(1, 2, gridspec_kw = {'width_ratios': [1, 4]})
    
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_label_position("right")
    
    for ax in axs:
        ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
        #ax.axhline(0.0, color = 'gray')
    
    axs[0].axvspan(-10, 0, alpha = 0.5, color = 'lightgray')
    
    axs[0].set_xlim(-5, -1)
    axs[0].set_ylim(-0.5, 6.5)
    
    axs[1].set_xlim(1, 60)
    axs[1].set_ylim(-0.2, 2.5)
    
    for i, ax in enumerate(axs):
        kwargs = {'model': '8 RF-D20-E5000', 'value_fn': TF_MEAN_FN, 
                  #'error_fn': TF_MEANERR_FN, 
                  'error_band': 0.9 }
        if i==0:
            kwargs['flip_neg_time'] = False
        plot_series(ax, results, feature_set = 'FE', label = 'FE', **kwargs)
        plot_series(ax, results, feature_set = 'VR', label = 'VR', **kwargs)
        plot_series(ax, results, feature_set = 'UIQ', label = 'UIQ', **kwargs)
        plot_series(ax, results, feature_set = 'POL+32', label = 'POL', **kwargs)
        
        kwargs.update({
                'model': '8 RF-D20-E5000', 'value_fn': TF_MEAN_FN, 
                #'error_fn': TF_MEANERR_FN, 
                'error_band': 0.9,
                })
        plot_series(ax, results, feature_set = 'FE+POL+UIQ+VR+54', 
                    label = 'all', color = 'k', **kwargs)
    
    #del kwargs['error_fn']
    #plot_series(axs[0], results, feature_set = 'DIS+FE+FR+POL+54', 
    #            label = 'FE+FR+DIS+POL', color = 'grey', **kwargs)
    
    axs[0].set_ylabel('Contemporary top-flop mean return', fontsize = 13)
    axs[1].set_ylabel('Post-event top-flop mean return', fontsize = 13)
    
    fig.text(0.14, 0.95, 'Contemporary', 
             horizontalalignment = 'center', verticalalignment = 'bottom', 
             transform = ax.transAxes, size = 13)
    fig.text(0.6, 0.95, 'Post-event', 
             horizontalalignment = 'center', verticalalignment = 'bottom', 
             transform = ax.transAxes, size = 13)
    
    axs[1].set_xlabel('Forecast horizon $T$', fontsize = 13, ha = 'right')
    axs[1].legend(loc = 'upper center', prop = {'size': 12}, ncol = 5, frameon = False)
    
    axs[0].set_xticks([-4, -2])
    axs[1].set_xticks([1, 10, 20, 30, 40, 50 , 60])
    
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.01, wspace = 0.02,
                        left = 0.05, right = 0.93,
                        bottom = 0.10, top = 0.94)
    
    fig.savefig(run_folder + 'time_dep_top-flop_mean.pdf')
    plt.close()



def plot_mean_total_return_fs(results, run_folder):
    fig = plt.figure(figsize = (10, 5))
    axs = fig.subplots(1, 2, gridspec_kw = {'width_ratios': [1, 4]})
    
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_label_position("right")
    
    for ax in axs:
        ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
        #ax.axhline(0.0, color = 'gray')
    
    axs[0].axvspan(-10, 0, alpha = 0.5, color = 'lightgray')
    
    axs[0].set_xlim(-5, -1)
    axs[0].set_ylim(-0.5, 3.0)
    
    axs[1].set_xlim(1, 60)
    axs[1].set_ylim(-0.2, 0.8)
    
    for i, ax in enumerate(axs):
        kwargs = {'model': '8 RF-D20-E5000', 'value_fn': MEAN_FN, 
                  #'error_fn': TF_MEANERR_FN, 
                  'error_band': 0.9 }
        if i==0:
            kwargs['flip_neg_time'] = False
        plot_series(ax, results, feature_set = 'FE', label = 'FE', **kwargs)
        plot_series(ax, results, feature_set = 'VR', label = 'VR', **kwargs)
        plot_series(ax, results, feature_set = 'UIQ', label = 'UIQ', **kwargs)
        plot_series(ax, results, feature_set = 'POL+32', label = 'POL', **kwargs)
        
        kwargs.update({
                'model': '8 RF-D20-E5000', 'value_fn': MEAN_FN, 
                #'error_fn': TF_MEANERR_FN, 
                'error_band': 0.9,
                })
        plot_series(ax, results, feature_set = 'FE+POL+UIQ+VR+54', 
                    label = 'all', color = 'k', **kwargs)
    
    #del kwargs['error_fn']
    #plot_series(axs[0], results, feature_set = 'DIS+FE+FR+POL+54', 
    #            label = 'FE+FR+DIS+POL', color = 'grey', **kwargs)
    
    axs[0].set_ylabel('Contemporary mean return', fontsize = 13)
    axs[1].set_ylabel('Post-event mean return', fontsize = 13)
    
    fig.text(0.14, 0.95, 'Contemporary', 
             horizontalalignment = 'center', verticalalignment = 'bottom', 
             transform = ax.transAxes, size = 13)
    fig.text(0.6, 0.95, 'Post-event', 
             horizontalalignment = 'center', verticalalignment = 'bottom', 
             transform = ax.transAxes, size = 13)
    
    axs[1].set_xlabel('Forecast horizon $T$', fontsize = 13, ha = 'right')
    axs[1].legend(loc = 'upper center', prop = {'size': 12}, ncol = 5, frameon = False)
    
    axs[0].set_xticks([-4, -2])
    axs[1].set_xticks([1, 10, 20, 30, 40, 50 , 60])
    
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.01, wspace = 0.02,
                        left = 0.08, right = 0.92,
                        bottom = 0.10, top = 0.94)
    
    fig.savefig(run_folder + 'time_dep_mean.pdf')
    plt.close()


def plot_mean_return_fs_combined(results, run_folder):
    fig = plt.figure(figsize = (11, 7))
    axs = fig.subplots(2, 2, 
                       gridspec_kw = {'width_ratios': [1, 4]})
    
    for ax in axs[:, 1]:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    
    for ax in axs.flatten():
        ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
        #ax.axhline(0.0, color = 'gray')
    
    
    for ax in axs[:, 0]:
        ax.set_xlim(-5, -1)
        ax.set_xticks([-4, -2])
        ax.axvspan(-10, 0, alpha = 0.5, color = 'lightgray')
    
    for ax in axs[:, 1]:
        ax.set_xlim(1, 60)
        ax.set_xticks([1, 10, 20, 30, 40, 50 , 60])
    
    for ax in axs[0, :]:
        ax.set_xticks([])
    
    axs[0, 0].set_ylim(-0.4, 3.1)
    axs[0, 0].set_yticks([0, 1, 2, 3])
    axs[0, 1].set_ylim(-0.1, 0.75)
    
    axs[1, 0].set_ylim(-0.4, 6.5)
    axs[1, 1].set_ylim(-0.1, 2.1)
    
    for i, ax in enumerate(axs.flatten()):
        kwargs = {'model': '8 RF-D20-E5000', 
                  'value_fn': TF_MEAN_FN if i >= 2 else MEAN_FN, 
                  #'error_fn': TF_MEANERR_FN, 
                  'error_band': 0.9 }
        if i % 2 == 0:
            kwargs['flip_neg_time'] = False
            
        for fs, label in [('FE', 'FE'), ('VR', 'VR'), ('UIQ', 'UIQ'), ('POL+32', 'POL')]:
            plot_series(ax, results, 
                        feature_set = fs, label = label, 
                        color = COLORS[label], linestyle = STYLES[label],
                        **kwargs)
        
        kwargs.update({
                'model': '8 RF-D20-E5000', 
                'value_fn': TF_MEAN_FN if i >= 2 else MEAN_FN, 
                #'error_fn': TF_MEANERR_FN, 
                'error_band': 0.9,
                })
        plot_series(ax, results, feature_set = 'FE+POL+UIQ+VR+54', 
                    label = 'all', color = 'k', **kwargs)
    
    #del kwargs['error_fn']
    #plot_series(axs[0], results, feature_set = 'DIS+FE+FR+POL+54', 
    #            label = 'FE+FR+DIS+POL', color = 'grey', **kwargs)
    
    axs[1, 0].set_ylabel('Top-flop mean return', fontsize = 13)
    axs[1, 1].set_ylabel('Top-flop mean return', fontsize = 13)
    
    axs[0, 0].set_ylabel('Mean return', fontsize = 13)
    axs[0, 1].set_ylabel('Mean return', fontsize = 13)
    
    
    fig.text(0.14, 0.97, 'Contemporary', 
             horizontalalignment = 'center', verticalalignment = 'bottom', 
             transform = ax.transAxes, size = 13)
    fig.text(0.6, 0.97, 'Post-event', 
             horizontalalignment = 'center', verticalalignment = 'bottom', 
             transform = ax.transAxes, size = 13)
    
    axs[1, 1].set_xlabel('Forecast horizon $T$', fontsize = 13, ha = 'right')
    axs[0, 1].legend(loc = 'upper center', prop = {'size': 12}, ncol = 5, frameon = False)
    
    
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.00, wspace = 0.02,
                        left = 0.05, right = 0.94,
                        bottom = 0.08, top = 0.96)
    
    fig.savefig(run_folder + 'time_dep_fs_mean_combined.pdf')
    plt.close()




def plot_mean_return_fs_value_glamor(run_folder, 
                                     alpha = 0.1, 
                                     value_glamor_cutoff = 0.1,
                                     event_type = 'all',
                                     class_feature = 'BM_ratio'):
    
    jobs = [f for f in os.listdir(run_folder) if f.startswith('job')]
    jobs.sort(key = lambda s: int(s.replace('job-', '')))
    
    SR = []
    
    for job in jobs:
        if not os.path.isfile(run_folder + '%s/model_results.hdf' % job):
            continue
        
        result = pd.read_hdf(run_folder + '%s/model_results.hdf' % job)
        info = extract_job_info(result)
        
        if info['model'] != '8 RF-D20-E5000':
            continue
        
        period = info['period']
        if period not in SEL_TIMES:
            continue
        
        feature_set = info['features']
        
        if not os.path.isfile(run_folder + '%s/predictions.hdf' % job):
            continue
        
        preds = pd.read_hdf(run_folder + '%s/predictions.hdf' % job)
        
        preds = preds.join(data.set_index(['ticker_symbol', 'fiscal_period', 'local_date'])[['CP_ratio', 'EP_ratio']],
                           on = ['ticker_symbol', 'fiscal_period', 'local_date'])
        
        preds['is_top_flop'] = (preds.pred_rank < alpha) | (preds.pred_rank > 1 - alpha)
        preds['bin'] = pd.qcut(preds[class_feature], 
                               [0.0, value_glamor_cutoff, 1.0 - value_glamor_cutoff, 1.0], 
                               labels = ['glamor', 'neutral', 'value'],
                               duplicates = 'drop')
        
        all_result = preds.groupby('bin')['signed_target_return'].agg(['mean', 'count'])
        all_result['mean'] *= 1E4
        all_result.columns = pd.MultiIndex.from_product([['all'], all_result.columns])
        
        s_preds = preds[preds['is_top_flop']]
        top_flop_result = s_preds.groupby('bin')['signed_target_return'].agg(['mean', 'count'])
        top_flop_result['mean'] *= 1E4
        top_flop_result.columns = pd.MultiIndex.from_product([['top_flop'], top_flop_result.columns])
        top_flop_result.index = all_result.index
        
        R = pd.concat([all_result, top_flop_result], axis = 1)
        
        R.index = R.index.astype(str)
        R = pd.concat([R], keys = [period], names = ['period'])
        R = pd.concat([R], keys = [feature_set], names = ['feature_set'])
        
        SR.append(R)
    
    SR = pd.concat(SR)
    
    
    fig = plt.figure(figsize = (11, 10))
    axs = fig.subplots(3, 2, 
                       gridspec_kw = {'width_ratios': [1, 4]})
    
    for ax in axs[:, 1]:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    
    for ax in axs.flatten():
        ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
    
    for ax in axs[:, 0]:
        ax.set_xlim(-5, -1)
        ax.set_xticks([-4, -2])
        ax.axvspan(-10, 0, alpha = 0.5, color = 'lightgray')
    
    for ax in axs[:, 1]:
        ax.set_xlim(1, 60)
        ax.set_xticks([1, 10, 20, 30, 40, 50 , 60])
    
    for ax in axs[0:2, :].flatten():
        ax.set_xticks([])
    
    for row in range(3):
        if event_type == 'all':
            axs[row, 0].set_ylim(-0.4, 3.5)
            axs[row, 0].set_yticks([0, 1, 2, 3])
            axs[row, 1].set_ylim(-0.1, 1.4)
        else:
            axs[row, 0].set_ylim(-0.4, 6.5)
            axs[row, 0].set_yticks([0, 1, 2, 3, 4, 5, 6])
            axs[row, 1].set_ylim(-0.1, 2.9)
    
    
    for vg_type_index, vg_type in enumerate(['value', 'neutral', 'glamor']):
        
        for cont_index, cont in enumerate([True, False]):
            ax = axs[vg_type_index, cont_index]
            
            SER = [('FE', 'FE'), ('VR', 'VR'), ('UIQ', 'UIQ'), ('POL+32', 'POL'), 
                   ('FE+POL+UIQ+VR+54', 'all')]
            
            for fs, label in SER:
                plt_data = SR.loc[(fs, slice(None), vg_type), (event_type, 'mean')]
                plt_data.sort_index(inplace = True)
                
                ax.plot(plt_data.index.get_level_values('period'), plt_data/100,
                        label = label, 
                        color = COLORS[label], linestyle = STYLES[label])
    
    
    axs[2, 0].set_ylabel('Glamor mean return', fontsize = 13)
    axs[2, 1].set_ylabel('Glamor mean return', fontsize = 13)
    
    axs[0, 0].set_ylabel('Value mean return', fontsize = 13)
    axs[0, 1].set_ylabel('Value mean return', fontsize = 13)
    
    axs[1, 0].set_ylabel('Neutral mean return', fontsize = 13)
    axs[1, 1].set_ylabel('Neutral mean return', fontsize = 13)
    
    fig.text(0.14, 0.97, 'Contemporary', 
             horizontalalignment = 'center', verticalalignment = 'bottom', 
             transform = ax.transAxes, size = 13)
    fig.text(0.6, 0.97, 'Post-event', 
             horizontalalignment = 'center', verticalalignment = 'bottom', 
             transform = ax.transAxes, size = 13)
    
    axs[2, 1].set_xlabel('Forecast horizon $T$', fontsize = 13, ha = 'right')
    axs[0, 1].legend(loc = 'upper center', prop = {'size': 12}, ncol = 5, frameon = False)
    
    
    fig.align_labels()
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.00, wspace = 0.02,
                        left = 0.05, right = 0.94,
                        bottom = 0.06, top = 0.96)
    
    fig.savefig(run_folder + 'time_dep_fs_mean_value_glamor_by_%s_%s_%.2f.pdf' % (
            class_feature, event_type, value_glamor_cutoff))
    plt.close()




def plot_mean_return_value_glamor_by_market(run_folder, 
                                            alpha = 0.1, 
                                            value_glamor_cutoff = 0.2,
                                            event_type = 'all',
                                            class_feature = 'BM_ratio'):
    
    jobs = [f for f in os.listdir(run_folder) if f.startswith('job')]
    jobs.sort(key = lambda s: int(s.replace('job-', '')))
    
    SR = []
    
    for job in jobs:
        if not os.path.isfile(run_folder + '%s/model_results.hdf' % job):
            continue
        
        result = pd.read_hdf(run_folder + '%s/model_results.hdf' % job)
        info = extract_job_info(result)
        
        if info['model'] != '8 RF-D20-E5000':
            continue
        
        period = info['period']
        if period not in SEL_TIMES:
            continue
        
        feature_set = info['features']
        
        if not os.path.isfile(run_folder + '%s/predictions.hdf' % job):
            continue
        
        preds = pd.read_hdf(run_folder + '%s/predictions.hdf' % job)
        
        preds['is_top_flop'] = (preds.pred_rank < alpha) | (preds.pred_rank > 1 - alpha)
        preds['bin'] = pd.qcut(preds[class_feature], 
                               [0.0, value_glamor_cutoff, 1.0 - value_glamor_cutoff, 1.0], 
                               labels = ['Low BM', 'Interm. BM', 'High BM'],
                               duplicates = 'drop')
        
        all_result = preds.groupby(['mkt_index', 'bin'])['signed_target_return'].agg(['mean', 'count'])
        all_result['mean'] *= 1E4
        all_result.columns = pd.MultiIndex.from_product([['all'], all_result.columns])
        
        s_preds = preds[preds['is_top_flop']]
        top_flop_result = s_preds.groupby(['mkt_index', 'bin'])['signed_target_return'].agg(['mean', 'count'])
        top_flop_result['mean'] *= 1E4
        top_flop_result.columns = pd.MultiIndex.from_product([['top_flop'], top_flop_result.columns])
        top_flop_result.index = all_result.index
        
        R = pd.concat([all_result, top_flop_result], axis = 1)
        
        R = pd.concat([R], keys = [period], names = ['period'])
        R = pd.concat([R], keys = [feature_set], names = ['feature_set'])
        
        SR.append(R)
    
    SR = pd.concat(SR)
    
    
    fig = plt.figure(figsize = (11, 8))
    axs = fig.subplots(3, 2, gridspec_kw = {'width_ratios': [1, 4]}, sharex = 'col')
    
    for i in range(3):
        axs[i, 1].yaxis.tick_right()
        axs[i, 1].yaxis.set_label_position("right")
        
        for ax in axs[i, :]:
            ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
        
        axs[i, 0].axvspan(-10, 0, alpha = 0.5, color = 'lightgray')
        axs[i, 1].axhline(0.0, color = 'gray')
        
        axs[i, 0].set_xlim(-5, -1)
        axs[i, 1].set_xlim(1, 60)
        axs[i, 1].set_xticks([1, 10, 20, 30, 40, 50, 60])
        
        if event_type == 'all':
            axs[i, 0].set_ylim(-0.4, 4.4)
            axs[i, 1].set_ylim(-0.4, 1.6)
        else:
            axs[i, 0].set_ylim(-0.4, 6.5)
            axs[i, 1].set_ylim(-0.4, 3)
        
    colors = ['blue', 'k', 'red']
    styles = ['dashed', 'solid', 'dotted']
    mkt_names = {'SP500TR': 'S&P500', 'SP400TR': 'S&P400', 'SP600TR': 'S&P600'}
    
    for vg_type_index, vg_type in enumerate(['Low BM', 'Interm. BM', 'High BM']):
        for mkt_index, mkt in enumerate(['SP500TR', 'SP400TR', 'SP600TR']):
            
            plt_data = SR.loc[('FE+POL+UIQ+VR+54', slice(None), mkt, vg_type), (event_type, 'mean')]
            plt_data.sort_index(inplace = True)
                    
            axs[mkt_index, 0].plot(plt_data.index.get_level_values('period'), plt_data/100,
                               color = colors[vg_type_index], linestyle = styles[vg_type_index],
                               label = vg_type)
            axs[mkt_index, 1].plot(plt_data.index.get_level_values('period'), plt_data/100,
                                color = colors[vg_type_index], linestyle = styles[vg_type_index],
                                label = vg_type)
            
            axs[mkt_index, 0].set_ylabel(mkt_names[mkt] + ' return', fontsize = 13)
            axs[mkt_index, 1].set_ylabel(mkt_names[mkt] + ' return', fontsize = 13)
    
    fig.text(0.14, 0.96, 'Contemporary', 
             horizontalalignment = 'center', verticalalignment = 'bottom', 
             transform = ax.transAxes, size = 13)
    fig.text(0.6, 0.96, 'Post-event', 
             horizontalalignment = 'center', verticalalignment = 'bottom', 
             transform = ax.transAxes, size = 13)
    
    axs[2, 1].set_xlabel('Forecast horizon $T$', fontsize = 13, ha = 'right')
    axs[0, 1].legend(loc = 'upper center', prop = {'size': 13}, ncol = 5, frameon = False)
    
    fig.align_labels()
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.00, wspace = 0.02, left = 0.05, right = 0.94,
                        bottom = 0.07, top = 0.96)
    
    fig.savefig(run_folder + 'time_dep_mean_value_glamor_by_market_by_%s_%s_%.2f.pdf' % (
            class_feature, event_type, value_glamor_cutoff))
    plt.close()
    
    
    SR[('top_flop', 'ratio')] = SR[('top_flop', 'count')] / SR[('all', 'count')]
    
    SR.index.set_levels(['2SP400TR', '1SP500TR', '3SP600TR'], level = 2, inplace = True)
    
    ESR = SR.loc[('FE+POL+UIQ+VR+54', [-5, 5, 20, 60], 
                  slice(None), slice(None)), 
                [('all', 'mean'), ('top_flop', 'ratio')]]
    
    ESR.index = ESR.index.droplevel(0)
    ESR = ESR.unstack(1).sort_index(axis = 1)
    
    ESR_L = ESR.to_latex(float_format = '%.4f')
    
    



def plot_mean_return_value_glamor(run_folder, 
                                  alpha = 0.1, 
                                  value_glamor_cutoff = 0.2,
                                  event_type = 'all',
                                  class_feature = 'BM_ratio'):
    
    jobs = [f for f in os.listdir(run_folder) if f.startswith('job')]
    jobs.sort(key = lambda s: int(s.replace('job-', '')))
    
    SR = []
    
    for job in jobs:
        if not os.path.isfile(run_folder + '%s/model_results.hdf' % job):
            continue
        
        result = pd.read_hdf(run_folder + '%s/model_results.hdf' % job)
        info = extract_job_info(result)
        
        if info['model'] != '8 RF-D20-E5000':
            continue
        
        period = info['period']
        if period not in SEL_TIMES:
            continue
        
        feature_set = info['features']
        
        if not os.path.isfile(run_folder + '%s/predictions.hdf' % job):
            continue
        
        preds = pd.read_hdf(run_folder + '%s/predictions.hdf' % job)
        
        preds['is_top_flop'] = (preds.pred_rank < alpha) | (preds.pred_rank > 1 - alpha)
        preds['bin'] = pd.qcut(preds[class_feature], 
                               [0.0, value_glamor_cutoff, 1.0 - value_glamor_cutoff, 1.0], 
                               labels = ['Low BM', 'Intermediate BM', 'High BM'],
                               duplicates = 'drop')
        
        all_result = preds.groupby('bin')['signed_target_return'].agg(['mean', 'count'])
        all_result['mean'] *= 1E4
        all_result.columns = pd.MultiIndex.from_product([['all'], all_result.columns])
        
        s_preds = preds[preds['is_top_flop']]
        top_flop_result = s_preds.groupby('bin')['signed_target_return'].agg(['mean', 'count'])
        top_flop_result['mean'] *= 1E4
        top_flop_result.columns = pd.MultiIndex.from_product([['top_flop'], top_flop_result.columns])
        top_flop_result.index = all_result.index
        
        R = pd.concat([all_result, top_flop_result], axis = 1)
        
        R.index = R.index.astype(str)
        R = pd.concat([R], keys = [period], names = ['period'])
        R = pd.concat([R], keys = [feature_set], names = ['feature_set'])
        
        SR.append(R)
    
    SR = pd.concat(SR)
    
    
    fig = plt.figure(figsize = (11, 4))
    axs = fig.subplots(1, 2, gridspec_kw = {'width_ratios': [1, 4]})
    
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_label_position("right")
    
    for ax in axs:
        ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
        #ax.axhline(0.0, color = 'gray')
    
    axs[0].axvspan(-10, 0, alpha = 0.5, color = 'lightgray')
    
    if event_type == 'all':
        axs[0].set_ylim(0, 2.9)
        axs[1].set_ylim(0, 1.1)
    else:
        axs[0].set_ylim(0, 6.8)
        axs[1].set_ylim(0, 2.7)
    
    axs[0].set_xlim(-5, -1)
    axs[1].set_xlim(1, 60)
    
    colors = ['blue', 'k', 'red']
    styles = ['dashed', 'solid', 'dotted']
    
    for vg_type_index, vg_type in enumerate(['Low BM', 'Intermediate BM', 'High BM']):
        
        plt_data = SR.loc[('FE+POL+UIQ+VR+54', slice(None), vg_type), (event_type, 'mean')]
        plt_data.sort_index(inplace = True)
                
        axs[0].plot(plt_data.index.get_level_values('period'), plt_data/100,
                    color = colors[vg_type_index], linestyle = styles[vg_type_index],
                    label = vg_type)
        axs[1].plot(plt_data.index.get_level_values('period'), plt_data/100,
                    color = colors[vg_type_index], linestyle = styles[vg_type_index],
                    label = vg_type)
    
    axs[0].set_ylabel('Mean return', fontsize = 13)
    axs[1].set_ylabel('Mean return', fontsize = 13)
    
    fig.text(0.14, 0.95, 'Contemporary', 
             horizontalalignment = 'center', verticalalignment = 'bottom', 
             transform = ax.transAxes, size = 13)
    fig.text(0.6, 0.95, 'Post-event', 
             horizontalalignment = 'center', verticalalignment = 'bottom', 
             transform = ax.transAxes, size = 13)
    
    axs[1].set_xlabel('Forecast horizon $T$', fontsize = 13, ha = 'right')
    axs[1].legend(loc = 'upper center', prop = {'size': 12}, ncol = 5, frameon = False)
    
    fig.align_labels()
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.00, wspace = 0.02,
                        left = 0.06, right = 0.94,
                        bottom = 0.12, top = 0.94)
    
    fig.savefig(run_folder + 'time_dep_mean_value_glamor_by_%s_%s_%.2f.pdf' % (
            class_feature, event_type, value_glamor_cutoff))
    plt.close()




def plot_mean_return_fs_market_segment(run_folder, 
                                     alpha = 0.1, 
                                     event_type = 'all'):
    
    jobs = [f for f in os.listdir(run_folder) if f.startswith('job')]
    jobs.sort(key = lambda s: int(s.replace('job-', '')))
    
    SR = []
    
    for job in jobs:
        if not os.path.isfile(run_folder + '%s/model_results.hdf' % job):
            continue
        
        result = pd.read_hdf(run_folder + '%s/model_results.hdf' % job)
        info = extract_job_info(result)
        
        if info['model'] != '8 RF-D20-E5000':
            continue
        
        period = info['period']
        if period not in SEL_TIMES:
            continue
        
        feature_set = info['features']
        
        if not os.path.isfile(run_folder + '%s/predictions.hdf' % job):
            continue
        
        preds = pd.read_hdf(run_folder + '%s/predictions.hdf' % job)
        
        preds['is_top_flop'] = (preds.pred_rank < alpha) | (preds.pred_rank > 1 - alpha)
        
        all_result = preds.groupby('mkt_index')['signed_target_return'].agg(['mean', 'count'])
        all_result['mean'] *= 1E4
        all_result.columns = pd.MultiIndex.from_product([['all'], all_result.columns])
        
        s_preds = preds[preds['is_top_flop']]
        top_flop_result = s_preds.groupby('mkt_index')['signed_target_return'].agg(['mean', 'count'])
        top_flop_result['mean'] *= 1E4
        top_flop_result.columns = pd.MultiIndex.from_product([['top_flop'], top_flop_result.columns])
        top_flop_result.index = all_result.index
        
        R = pd.concat([all_result, top_flop_result], axis = 1)
        
        R.index = R.index.astype(str)
        R = pd.concat([R], keys = [period], names = ['period'])
        R = pd.concat([R], keys = [feature_set], names = ['feature_set'])
        
        SR.append(R)
    
    SR = pd.concat(SR)
    
    
    fig = plt.figure(figsize = (11, 10))
    axs = fig.subplots(3, 2, 
                       gridspec_kw = {'width_ratios': [1, 4]})
    
    for ax in axs[:, 1]:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    
    for ax in axs.flatten():
        ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
    
    for ax in axs[:, 0]:
        ax.set_xlim(-5, -1)
        ax.set_xticks([-4, -2])
        ax.axvspan(-10, 0, alpha = 0.5, color = 'lightgray')
    
    for ax in axs[:, 1]:
        ax.set_xlim(1, 60)
        ax.set_xticks([1, 10, 20, 30, 40, 50 , 60])
    
    for ax in axs[0:2, :].flatten():
        ax.set_xticks([])
    
    for row in range(3):
        if event_type == 'all':
            axs[row, 0].set_ylim(-0.4, 3.5)
            axs[row, 0].set_yticks([0, 1, 2, 3])
            axs[row, 1].set_ylim(-0.1, 1.4)
        else:
            axs[row, 0].set_ylim(-0.4, 6.5)
            axs[row, 0].set_yticks([0, 1, 2, 3, 4, 5, 6])
            axs[row, 1].set_ylim(-0.1, 2.9)
    
    
    for vg_type_index, vg_type in enumerate(['SP500TR', 'SP600TR', 'SP400TR']):
        
        for cont_index, cont in enumerate([True, False]):
            ax = axs[vg_type_index, cont_index]
            
            SER = [('FE', 'FE'), ('VR', 'VR'), ('UIQ', 'UIQ'), ('POL+32', 'POL'), 
                   ('FE+POL+UIQ+VR+54', 'all')]
            
            for fs, label in SER:
                plt_data = SR.loc[(fs, slice(None), vg_type), (event_type, 'mean')]
                plt_data.sort_index(inplace = True)
                
                ax.plot(plt_data.index.get_level_values('period'), plt_data/100,
                        label = label, 
                        color = COLORS[label], linestyle = STYLES[label])
    
    
    axs[2, 0].set_ylabel('SP600 mean return', fontsize = 13)
    axs[2, 1].set_ylabel('SP600 mean return', fontsize = 13)
    
    axs[0, 0].set_ylabel('SP500 mean return', fontsize = 13)
    axs[0, 1].set_ylabel('SP500 mean return', fontsize = 13)
    
    axs[1, 0].set_ylabel('SP400 mean return', fontsize = 13)
    axs[1, 1].set_ylabel('SP400 mean return', fontsize = 13)
    
    fig.text(0.14, 0.97, 'Contemporary', 
             horizontalalignment = 'center', verticalalignment = 'bottom', 
             transform = ax.transAxes, size = 13)
    fig.text(0.6, 0.97, 'Post-event', 
             horizontalalignment = 'center', verticalalignment = 'bottom', 
             transform = ax.transAxes, size = 13)
    
    axs[2, 1].set_xlabel('Forecast horizon $T$', fontsize = 13, ha = 'right')
    axs[0, 1].legend(loc = 'upper center', prop = {'size': 12}, ncol = 5, frameon = False)
    
    
    fig.align_labels()
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.00, wspace = 0.02,
                        left = 0.05, right = 0.94,
                        bottom = 0.06, top = 0.96)
    
    fig.savefig(run_folder + 'time_dep_fs_mean_by_market_segment_%s.pdf' % (
                    event_type))
    plt.close()




def plot_feature_imp(results, run_folder, 
                     model = '8 RF-D20-E5000', feature_importance_type = 'feature_imp_gini'):
    I = list(map(analyze_feature_importance, results.to_dict(orient = 'records')))
    I = pd.concat([imps[feature_importance_type] for imps in I], axis = 1).transpose().set_index(results.index)
    
    SP = [#('FE+FR', 'FE+FR'), #('DIS+FE+FR', 'DIS+FE+FR'), 
          ('FE', 'FE'), ('VR', 'VR'), 
          ('POL+32', 'POL'),
          ('FE+POL+UIQ+VR+54', 'all')]
    for fs, fl in SP:
        SI = (I.index.get_level_values('features') == fs) & \
                (I.index.get_level_values('model') == model)
        SI = I.loc[SI].sort_index().dropna(axis = 1)
        
        Cs = [(f, s) for s in FEATURE_SETS_ORDER for f in FEATURE_SETS_DICT[s] if s in fs.split('+')]
        Cs = [(c, s) for (c, s) in Cs if c in SI.columns]
        
        fig = plt.figure(figsize = (14, 8))
        ax = fig.subplots(1, 1)
        
        GI = []
        for fsg in FEATURE_SETS_ORDER:
            f = FEATURE_SETS_DICT[fsg]
            
            if len(set(f).intersection(SI.columns)) != len(f):
                continue
            
            GI.append(pd.Series(SI[f].sum(axis = 1), name = fsg))
        GI = pd.concat(GI, axis = 1)
        
        plot_cols = [c for c, s in Cs]
        
        ax.stackplot(GI.index.get_level_values('period'), GI.as_matrix().transpose())
        
        ax.stackplot(SI[plot_cols].index.get_level_values('period'), 
                     SI[plot_cols].as_matrix().transpose(), 
                     edgecolor = 'k', colors = 'white', alpha = 0.5)
        
        last_frac = SI[plot_cols].iloc[-1]
        cur_frac = 0.0
        for fn in plot_cols:
            next_cur_frac = cur_frac + last_frac[fn]
            if last_frac[fn] > 0.02:
                text = FEATURE_NAME_DICT[fn] if fn in FEATURE_NAME_DICT else fn
                text = text.replace('_LM', '')
                ax.text(1.01, (cur_frac + next_cur_frac)/2 - 0.005, text, 
                        verticalalignment = 'center', transform = (ax.transAxes), 
                        fontsize = 13)
            cur_frac = next_cur_frac
        
        first_frac = SI[plot_cols].iloc[0]
        cur_frac = 0.0
        for fn in plot_cols:
            next_cur_frac = cur_frac + first_frac[fn]
            if first_frac[fn] > 0.02:
                text = FEATURE_NAME_DICT[fn] if fn in FEATURE_NAME_DICT else fn
                text = text.replace('_LM', '')
                ax.text(-0.01, (cur_frac + next_cur_frac)/2 - 0.005, text, 
                        verticalalignment = 'center', horizontalalignment = 'right',
                        transform = (ax.transAxes), 
                        fontsize = 13)
            cur_frac = next_cur_frac
            
        ax.set_xlim(-5, 60)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
        ax.set_xlabel("event time", fontsize = 13)
        
        ax.axes.get_yaxis().set_visible(False)
        
        fig.tight_layout()
        fig.subplots_adjust(hspace = 0.0, wspace = 0.0,
                            left = 0.08, right = 0.92,
                            bottom = 0.07, top = 0.99)
        
        fig.savefig(run_folder + '/time_dep_feature_imp-%s.pdf' % fs)
        plt.close()



def plot_feature_imp_2(results, run_folder, 
                       sel_times = [-5, -4, -3, -2, -1, 1, 2, 3, 5, 10, 20, 30, 40, 50, 60], 
                       model = '8 RF-D20-E5000', feature_importance_type = 'feature_imp_gini'):
    I = list(map(analyze_feature_importance, results.to_dict(orient = 'records')))
    I = pd.concat([imps[feature_importance_type] for imps in I], axis = 1).transpose().set_index(results.index)
    
    OFFSETS = {'VR': 0.17, 'FE': 0.45, 'UIQ': 0.7, 'POL': 1.05}
    
    SP = [#('FE+FR', 'FE+FR'), #('DIS+FE+FR', 'DIS+FE+FR'), 
          #('FE', 'FE'), ('VR', 'VR'), 
          #('POL+32', 'POL'),
          ('FE+POL+UIQ+VR+54', 'all')]
    for fs, fl in SP:
        SI = (I.index.get_level_values('features') == fs) & \
                (I.index.get_level_values('model') == model) & \
                    I.index.get_level_values('period').isin(sel_times)
        SI = I.loc[SI].sort_index().dropna(axis = 1)
        
        Cs = [(f, s) for s in FEATURE_SETS_ORDER for f in FEATURE_SETS_DICT[s] if s in fs.split('+')]
        Cs = [(c, s) for (c, s) in Cs if c in SI.columns]
        
        fig = plt.figure(figsize = (12, 10))
        ax = fig.subplots(1, 1)
        
        #rect = patches.Rectangle((50, 100), 40, 30,
        #                         linewidth=1, edgecolor='r', facecolor='gray')
        #ax.add_patch(rect)
        #ax.axvline(0, color = 'gray', alpha = 0.5, zorder = -1)
        
        GI = []
        for fsg in FEATURE_SETS_ORDER:
            f = FEATURE_SETS_DICT[fsg]
            
            if len(set(f).intersection(SI.columns)) != len(f):
                continue
            
            GI.append(pd.Series(SI[f].sum(axis = 1), name = fsg))
        GI = pd.concat(GI, axis = 1)
        
        
        cur_line_offset = None
        for fg, offset in OFFSETS.items():
            ax.fill_between(GI.index.get_level_values('period'), 
                            offset - GI.loc[:, fg] / 2, offset + GI.loc[:, fg] / 2,
                            alpha = 0.9, linewidth = 0,
                            color = COLORS[fg])
            
            cur_line_offset = offset - GI.loc[:, fg] / 2
            
            #ax.plot(GI.index.get_level_values('period'), cur_line_offset,
            #        color = 'dimgray', linewidth = 1.5)
            
            for cf, cfg in Cs:
                if cfg != fg:
                    continue
                
                cur_line_offset += SI.loc[:, cf]
                
                if cf is not Cs[-1][0]:
                    ax.plot(GI.index.get_level_values('period'), cur_line_offset,
                            color = 'white', linewidth = 1.0)
                
                if SI.loc[(slice(None), -5, slice(None)), cf].max() > 0.018:
                    text = FEATURE_NAME_DICT[cf] if cf in FEATURE_NAME_DICT else cf
                    
                    fi = SI.loc[(slice(None), -5, slice(None)), cf].max()
                    co = cur_line_offset.loc[(slice(None), -5, slice(None))].mean()
                    
                    ax.text(-5.5, co - fi/2 - 0.005, text, 
                            verticalalignment = 'center', horizontalalignment = 'right',
                            transform = ax.transData, 
                            fontsize = 13)
                
                if SI.loc[(slice(None), 60, slice(None)), cf].max() > 0.018:
                    text = FEATURE_NAME_DICT[cf] if cf in FEATURE_NAME_DICT else cf
                    
                    fi = SI.loc[(slice(None), 60, slice(None)), cf].max()
                    co = cur_line_offset.loc[(slice(None), 60, slice(None))].mean()
                    
                    ax.text(60.5, co - fi/2 - 0.005, text, 
                            verticalalignment = 'center', horizontalalignment = 'left',
                            transform = ax.transData, 
                            fontsize = 13)
        
        ax.set_xlim(-5, 60)
        ax.set_ylim(0.01, 1.30)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
        ax.set_xlabel("Forecast horizon $T$", fontsize = 13)
        
        ax.axes.get_yaxis().set_visible(False)
        
        fig.tight_layout()
        fig.subplots_adjust(hspace = 0.0, wspace = 0.0,
                            left = 0.075, right = 0.92,
                            bottom = 0.05, top = 0.99)
        
        fig.savefig(run_folder + '/time_dep_feature_imp_2-%s.pdf' % fs)
        plt.close()



if __name__ == '__main__':
    results = read_run(RUN_FOLDER)
    
    plot_mean_return(results, RUN_FOLDER)
    plot_mean_return_fs(results, RUN_FOLDER)
    plot_mean_total_return_fs(results, RUN_FOLDER)
    plot_mean_return_fs_combined(results, RUN_FOLDER)
    plot_decile_returns_acc(results, RUN_FOLDER)
    
    plot_feature_imp(results, RUN_FOLDER)
    plot_feature_imp_2(results, RUN_FOLDER)
    
    for cutoff in [0.1, 0.2, 0.3]:
        for event_type in ['all', 'top_flop']:
            plot_mean_return_value_glamor(RUN_FOLDER, value_glamor_cutoff = cutoff, 
                                          event_type = event_type)
            plot_mean_return_value_glamor_by_market(RUN_FOLDER, value_glamor_cutoff = cutoff, 
                                          event_type = event_type)
    
    for cutoff in [0.1, 0.2, 0.3]:
        for event_type in ['all', 'top_flop']:
            for col in ['BM_ratio', 'CP_ratio', 'EP_ratio']:
                plot_mean_return_fs_value_glamor(RUN_FOLDER, 
                                                 value_glamor_cutoff = cutoff,
                                                 event_type = event_type,
                                                 class_feature = col)
                
                
