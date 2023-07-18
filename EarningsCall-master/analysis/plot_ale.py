#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 09:25:01 2020

@author: mschnaubelt
"""

import os
import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from analysis.analysis_helper import extract_job_info
from config import FINAL_ALE_RUN, FINAL_ALE_2D_RUN, FEATURE_NAME_DICT



ALE_SCALE_OVERRIDE = {
        'EP_surprise': (),
        'EP_ratio': ()
        }


def compute_weighted_plot_ale(ale, cut_quantiles, weights):
    plot_ale = ale.iloc[cut_quantiles:-cut_quantiles].copy()
    
    if weights is None:
        weights = [1 for i in range(7)]
    
    for i in range(7):
        plot_ale[i] -= plot_ale[i].mean()
        plot_ale['W_%d' % i] = plot_ale[i] * weights[i]
    
    plot_ale['mean_ale'] = plot_ale[['W_%d' % i for i in range(7)]].sum(axis = 1) / sum(weights)
    
    y2 = plot_ale[list(range(7))].sub(plot_ale[list(range(7))].mean(axis = 1), axis = 0)**2
    for i in range(7):
        y2[i] *= weights[i]
    
    plot_ale['std_ale'] = np.sqrt(y2.sum(axis = 1) / sum(weights))
    
    return plot_ale



def plot_1d_ale(ax, feature, ale, 
                show_quantile_labels = True, cut_quantiles = 2,
                weights = None,
                x_min = None, x_max = None,
                min_value = None, max_value = None):
    
    plot_ale = compute_weighted_plot_ale(ale, cut_quantiles, weights)
    
    x_values = (plot_ale['lower_value'] + plot_ale['upper_value']) / 2
    
    x_min = x_values.iloc[0] if x_min is None else x_min
    x_max = x_values.iloc[-1] if x_max is None else x_max
    y_min = plot_ale['mean_ale'].min() if min_value is None else min_value
    y_max = plot_ale['mean_ale'].max() if max_value is None else max_value
    
    ax.axhline(0, color = 'gray', linewidth = 0.75, alpha = 1.0)
    
    q_spacing = int(0.1 * len(ale))
    
    qs = ale['lower_value'].iloc[0::q_spacing]
    for i, q in enumerate(qs):
        ax.axvline(q, color = 'gray', linewidth = 0.75, alpha = 0.5, linestyle = ':')
        
        if (q > x_min) and (q < x_max) and show_quantile_labels:
            if ale.loc[qs.index[i]]['mean_ale'] < (y_max + y_min) / 2:
                yf = 0.9 + (i%4==1) * 0.07
            else:
                yf = 0.07 + (i%4==1) * 0.07
            
            if i % 2 == 0:
                continue
            
            bbox_props = dict(boxstyle = "round,pad=0.1", fc = "#ffffffd8", lw=0)
            
            ax.annotate('%.1f' % (i/10), xy = (q, yf), 
                        xycoords = ('data', 'axes fraction'),
                        ha = 'center', va = 'top', bbox = bbox_props,
                        color = 'darkgray', backgroundcolor = 'white')
    
    ax.plot(x_values, 
            plot_ale['mean_ale'] * 100, 
            label = 'mean')
    
    ax.fill_between(x_values, 
                    (plot_ale['mean_ale'] - plot_ale['std_ale'])*100,
                    (plot_ale['mean_ale'] + plot_ale['std_ale'])*100, 
                    alpha = 0.75, color = '#5799c6', linewidth = 0.0)
    
    ax.set_xlim(x_min, x_max)
    
    if (min_value is not None) and (max_value is not None):
        ax.set_ylim(min_value*100, max_value*100)



def plot_single_1d_ale(feature, ale, N_train, output_folder, 
                       min_value = None, max_value = None):
    fig = plt.figure(figsize = (6, 4))
    ax = fig.subplots(1, 1)
    
    plot_1d_ale(ax, feature, ale, weights = N_train, min_value = min_value, max_value = max_value)
    
    ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
    ax.set_xlabel(FEATURE_NAME_DICT[feature] if feature in FEATURE_NAME_DICT else feature, size = 13)
    
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.0, wspace = 0.0,
                        left = 0.12, right = 0.98,
                        bottom = 0.13, top = 0.98)
    
    fig.savefig(output_folder + 'ALE_1D-%s.pdf' % feature)
    plt.close()


def plot_horizons_1d_ale(feature, ales, horizons, N_train, output_folder):
    fig = plt.figure(figsize = (12, 3.1))
    axs = fig.subplots(1, len(ales), sharey = True)
    
    
    y_min, y_max = 100, -100
    for i, ale in enumerate(ales):
        plot_ale = compute_weighted_plot_ale(ale, 2, N_train)
        
        y_min = min((plot_ale['mean_ale'] - plot_ale['std_ale']).min(), y_min)
        y_max = max((plot_ale['mean_ale'] + plot_ale['std_ale']).max(), y_max)
        #print(feature, y_min, y_max)
    
    
    for i, ale in enumerate(ales):
        plot_1d_ale(axs[i], feature, ale, weights = N_train, 
                    min_value = min(y_min, -y_max), max_value = max(y_max, -y_min))
        
        axs[i].tick_params(axis = 'both', which = 'major', labelsize = 13)
        
        axs[i].text(0.90, 0.03, horizons[i], #ha = 'center', va = 'bottom',
                    transform = axs[i].transAxes, fontsize = 13)
        #axs[i].set_xlabel(FEATURE_NAME_DICT[feature] if feature in FEATURE_NAME_DICT else feature, size = 12)
    
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.0, wspace = 0.0,
                        left = 0.06, right = 0.99,
                        bottom = 0.09, top = 0.97)
    
    fig.savefig(output_folder + 'ALE_HOR_1D-%s.pdf' % feature)
    plt.close()



def plot_earnings_sales_1d_ale(data, output_folder):
    features = ['EP_surprise', 'EP_ratio', 'SP_surprise', 'SP_ratio']
    N_train = data[0]['train_lengths']
    
    X_LIMITS = {
            'EP_surprise': (-0.006, 0.006, [-0.004, 0.0, 0.004]),
            'EP_ratio': (-0.002, 0.036, [0.0, 0.01, 0.02, 0.03]),
            'SP_surprise': (-0.02, 0.02, [-0.015, 0.0, 0.015]),
            'SP_ratio': (0.05, 0.75, [0.1, 0.3, 0.5, 0.7])
            }
    
    fig = plt.figure(figsize = (12, 10))
    axs = fig.subplots(4, 4)
    
    for col, f in enumerate(features):
        for row, p in enumerate([-5, 5, 20, 60]):
            ales = [d['ale1d'] for d in data if d['period'] == p][-1]
            ale = ales[f]
            
            ax = axs[row, col]
            
            limit_val = 0.044 if row == 0 else 0.012
            
            plot_1d_ale(ax, f, ale, weights = N_train, 
                        x_min = X_LIMITS[f][0], x_max = X_LIMITS[f][1], 
                        min_value = -limit_val, max_value = limit_val)
            
            ax.yaxis.tick_right()
            ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
            
            if col != 3:
                ax.set_yticks([])
            
            if row != 3:
                ax.set_xticks([])
            
            if row == 3:
                ax.set_xlabel(FEATURE_NAME_DICT[f], fontsize = 13)
                ax.set_xticks(X_LIMITS[f][2])
            
            if col == 0:
                ax.text(-0.28, 0.5, "$T=%d$" % p, ha = 'left', va = 'center',
                        transform = ax.transAxes, fontsize = 13)
            
            if col == 3 and row > 0:
                ax.set_yticks([-1, 0, 1])
    
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.0, wspace = 0.0,
                        left = 0.07, right = 0.96,
                        bottom = 0.06, top = 0.99)
    
    fig.savefig(output_folder + 'EP_SP_ale.pdf')
    plt.close()



def plot_call_related_1d_ale(data, output_folder):
    features = ['log_length_intro', 'First_Ana_Pos', 'Last_Ana_Pos', 'environment_sq_LM_Negativity']
    N_train = data[0]['train_lengths']
    
    X_LIMITS = {
            'log_length_intro': (-0.006, 0.006, [-0.004, 0.0, 0.004]),
            'First_Ana_Pos': (-0.002, 0.036, [0.0, 0.01, 0.02, 0.03]),
            'Last_Ana_Pos': (-0.02, 0.02, [-0.015, 0.0, 0.015]),
            'environment_sq_LM_Negativity': (0.05, 0.75, [0.1, 0.3, 0.5, 0.7])
            }
    
    fig = plt.figure(figsize = (12, 7))
    axs = fig.subplots(3, len(features))
    
    for col, f in enumerate(features):
        for row, p in enumerate([-5, 5, 60]):
            ales = [d['ale1d'] for d in data if d['period'] == p][-1]
            ale = ales[f]
            
            ax = axs[row, col]
            
            limit_val = {0: 0.008, 1: 0.0013, 2: 0.0055}[row]
            tick_vals = {0: [-0.5, 0, 0.5], 1: [-0.1, 0, 0.1], 2: [-0.4, 0, 0.4]}[row]
            
            plot_1d_ale(ax, f, ale, weights = N_train, 
                        #x_min = X_LIMITS[f][0], x_max = X_LIMITS[f][1], 
                        min_value = -limit_val, max_value = limit_val)
            
            ax.yaxis.tick_right()
            ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
            
            if col != 3:
                ax.set_yticks([])
            
            if row != 2:
                ax.set_xticks([])
            
            if row == 2:
                ax.set_xlabel(FEATURE_NAME_DICT[f], fontsize = 13)
                #ax.set_xticks(X_LIMITS[f][2])
            
            if col == 0:
                ax.text(-0.28, 0.5, "$T=%d$" % p, ha = 'left', va = 'center',
                        transform = ax.transAxes, fontsize = 13)
            
            ax.set_yticks(tick_vals)
    
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.0, wspace = 0.0,
                        left = 0.07, right = 0.95,
                        bottom = 0.08, top = 0.99)
    
    fig.savefig(output_folder + 'Call-related_ale.pdf')
    plt.close()



def plot_valuation_related_1d_ale(data, output_folder):
    features = ['MV_log', 'BM_ratio', 'BM_surprise', 'CP_ratio', 'CP_surprise']
    N_train = data[0]['train_lengths']
    
    X_LIMITS = {
            'MV_log': (8.53, 10.8, [9.0, 9.5, 10.0, 10.5]),
            'BM_ratio': (0.08, 1.35, [0.25, 0.5, 0.75, 1.0, 1.25]),
            'BM_surprise': (-0.058, 0.058, [-0.04, 0.0, 0.04]),
            'CP_ratio': (-0.009, 0.09, [0.0, 0.025, 0.05, 0.075]),
            'CP_surprise': (-0.025, 0.025, [-0.02, 0.0, 0.02]),
            }
    
    fig = plt.figure(figsize = (14, 10))
    axs = fig.subplots(4, 5)
    
    for col, f in enumerate(features):
        for row, p in enumerate([-5, 5, 20, 60]):
            ales = [d['ale1d'] for d in data if d['period'] == p][-1]
            ale = ales[f]
            
            ax = axs[row, col]
            
            limit_val = 0.012 if row == 0 else 0.012
            
            plot_1d_ale(ax, f, ale, weights = N_train, 
                        x_min = X_LIMITS[f][0], x_max = X_LIMITS[f][1], 
                        min_value = -limit_val, max_value = limit_val)
            
            ax.yaxis.tick_right()
            ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
            
            if col != 4:
                ax.set_yticks([])
            
            if row != 3:
                ax.set_xticks([])
            
            if row == 3:
                ax.set_xlabel(FEATURE_NAME_DICT[f], fontsize = 13)
                ax.set_xticks(X_LIMITS[f][2])
            
            if col == 0:
                ax.text(-0.28, 0.5, "$T=%d$" % p, ha = 'left', va = 'center',
                        transform = ax.transAxes, fontsize = 13)
            
            if col == 4:
                ax.set_yticks([-1, 0, 1])
    
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.0, wspace = 0.0,
                        left = 0.06, right = 0.97,
                        bottom = 0.06, top = 0.99)
    
    fig.savefig(output_folder + 'Valuation_ale.pdf')
    plt.close()



def plot_1d_ale_transposed(ax, feature, ale, 
                           show_quantile_labels = True, cut_quantiles = 2):
    
    plot_ale = compute_weighted_plot_ale(ale, cut_quantiles, None)
    
    x_values = (plot_ale['lower_value'] + plot_ale['upper_value']) / 2
    
    x_min, x_max = x_values.iloc[0], x_values.iloc[-1]
    #y_min = ale.iloc[cut_quantiles:-cut_quantiles]['mean_ale'].min(), 
    #y_max = ale.iloc[cut_quantiles:-cut_quantiles]['mean_ale'].max()
    
    ax.axvline(0, color = 'gray', linewidth = 0.75, alpha = 1.0)
    
    qs = ale['lower_value'].iloc[0::5]
    for i, q in enumerate(qs):
        ax.axhline(q, color = 'gray', linewidth = 0.75, alpha = 0.5, linestyle = ':')
        
    
    ax.plot(plot_ale['mean_ale']*100, 
            x_values, 
            label = 'mean')
    
    ax.fill_betweenx(x_values, 
                     (plot_ale['mean_ale'] - plot_ale['std_ale'])*100,
                     (plot_ale['mean_ale'] + plot_ale['std_ale'])*100, 
                     alpha = 0.75, color = '#5799c6', linewidth = 0.0)
        
    ax.set_ylim(x_min, x_max)


def plot_2d_ale(fig, ax, first_feature, second_feature, ale2d, 
                cbar_ax = None, scale_colors = True):
    mean_ale = ale2d['mean_ale']
    mean_ale = mean_ale.loc[~mean_ale.index.duplicated(keep = 'first')]
    mean_ale = mean_ale.unstack()
    #mean_ale = ales[4].unstack()
    #mean_ale = mean_ale.iloc[1:-1, 1:-1]
    
    x_grid = mean_ale.columns.values
    x_grid = np.insert(x_grid, 0, x_grid[0] - (x_grid[1]-x_grid[0]))
    
    y_grid = mean_ale.index.values
    y_grid = np.insert(y_grid, 0, y_grid[0] - (y_grid[1]-y_grid[0]))
    
    vlim = np.round(mean_ale.abs().max().max() *100 * 1.25 + 0.1, 2)
    
    pcm = ax.pcolormesh(x_grid, y_grid, mean_ale.values * 100, 
                        cmap = 'bwr_r', 
                        vmin = - vlim if scale_colors else None, 
                        vmax = vlim if scale_colors else None)
    cbar = fig.colorbar(pcm, ax = ax, cax = cbar_ax, orientation='horizontal')
    
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    
    return cbar



def plot_single_1d_2d_ale(first_feature, second_feature, ale1d_first, ale1d_second, 
                          ale2d, output_folder):
    fig = plt.figure(figsize = (11, 6))
    axs = fig.subplots(2, 2, 
                       #sharex = 'row', sharey = 'col',
                       gridspec_kw = {'width_ratios': [4, 1], 'height_ratios': [3, 1]})
    
    axs[0, 0].get_shared_x_axes().join(axs[0, 0], axs[1, 0])
    axs[0, 0].get_shared_y_axes().join(axs[0, 0], axs[0, 1])
    
    fig.delaxes(axs[1, 1])
    #cax = plt.axes([0.8, 0.08, 0.03, 0.2])
    cax = plt.axes([0.58, 0.38, 0.15, 0.02])
    
    
    cbar = plot_2d_ale(fig, axs[0, 0], first_feature, second_feature, ale2d, 
                       cbar_ax = cax, scale_colors = False)
    plot_1d_ale(axs[1, 0], second_feature, ale1d_second, 
                show_quantile_labels = False, cut_quantiles = 1)
    plot_1d_ale_transposed(axs[0, 1], first_feature, ale1d_first, 
                           show_quantile_labels = False, cut_quantiles = 1)
    
    
    axs[0, 0].xaxis.label.set_visible(False)
    axs[0, 0].xaxis.tick_top()
    axs[0, 0].tick_params(axis = 'both', which = 'major', 
                           top = False, left = False, 
                           labeltop = False, labelleft = False)
    
    axs[0, 0].yaxis.label.set_visible(False)
    
    axs[0, 1].yaxis.tick_right()
    axs[0, 1].yaxis.set_label_position('right')
    #axs[0, 1].xaxis.tick_top()
    
    axs[0, 1].set_ylabel(FEATURE_NAME_DICT[first_feature] if first_feature in FEATURE_NAME_DICT else first_feature, 
                  size = 13)
    axs[1, 0].set_xlabel(FEATURE_NAME_DICT[second_feature] if second_feature in FEATURE_NAME_DICT else second_feature, 
                  size = 13)
    
    axs[0, 1].xaxis.set_major_locator(plt.MaxNLocator(3, symmetric = True))
    axs[1, 0].yaxis.set_major_locator(plt.MaxNLocator(3, symmetric = True))
    
    axs[1, 0].yaxis.tick_right()
    axs[1, 0].yaxis.set_label_position('right')
    
    #cax.xaxis.set_major_locator(plt.MaxNLocator(3, symmetric = True))
    
    #p_lim = max(np.abs(axs[0, 1].get_xlim()).max(), np.abs(axs[1, 0].get_ylim()).max())
    f1_lim = max(np.abs(axs[0, 1].get_xlim()))
    axs[0, 1].set_xlim(-f1_lim, f1_lim)
    
    f2_lim = max(np.abs(axs[1, 0].get_ylim()))
    axs[1, 0].set_ylim(-f2_lim, f2_lim)
    
    
    if first_feature == 'EP_surprise' and second_feature == 'SP_surprise':
        axs[1, 0].set_ylim(-4.4, 4.4) # y scale SPFE
        axs[0, 1].set_xlim(-4.4, 4.4) # x scale EPFE
        
        #prise': (-0.006, 0.006, [-0.004, 0.0, 0.004]),
        #    'SP_surprise': (-0.02, 0.02, [-0.015, 0.0, 0.015]),
        
        axs[1, 0].set_xlim(-0.005, 0.005) # x scale SPFE
        axs[0, 1].set_ylim(-0.0052, 0.0052) # y scale EPFE
    
    
    for ax in [axs[1, 0], axs[0, 1]]:
        ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
    cax.tick_params(axis = 'both', which = 'major', labelsize = 13)
    
    #axs[0, 1].tick_params(axis = 'y', which = 'major', rotation = 90, ha = 'center')
    plt.setp(axs[0, 1].get_yticklabels(), rotation = 90, ha="center", va = 'top', rotation_mode="anchor")
    
    cbar.set_ticks([-1, 0.0, 1])
    
    #fig.tight_layout()
    fig.subplots_adjust(hspace = 0.02, wspace = 0.02,
                        left = 0.01, right = 0.95,
                        bottom = 0.09, top = 0.99)
    fig.savefig(output_folder + 'ALE_1D_2D-%s:%s.pdf' % (first_feature, second_feature))
    plt.close()


def plot_single_2d_ale(first_feature, second_feature, ale2d, output_folder):
    fig = plt.figure(figsize = (6, 6))
    ax = fig.subplots(1, 1)
    
    plot_2d_ale(fig, ax, first_feature, second_feature, ale2d)
    
    ax.set_ylabel(FEATURE_NAME_DICT[first_feature] if first_feature in FEATURE_NAME_DICT else first_feature, 
                  size = 12)
    ax.set_xlabel(FEATURE_NAME_DICT[second_feature] if second_feature in FEATURE_NAME_DICT else second_feature, 
                  size = 12)
    
    ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax.tick_params(axis = 'y', which = 'major', labelsize = 12, rotation = 0)
    
    #fig.tight_layout()
    fig.subplots_adjust(hspace = 0.0, wspace = 0.0,
                        left = 0.16, right = 0.98,
                        bottom = 0.15, top = 0.98)
    fig.savefig(output_folder + 'ALE_2D-%s:%s.pdf' % (first_feature, second_feature))
    plt.close()



def plot_all_1d_ale(ale_dict, output_folder, min_value = None, max_value = None):
    N_train = ale_dict['train_lengths']
    
    for feature, ale in ale_dict['ale1d'].items():
        plot_single_1d_ale(feature, ale, N_train, output_folder, min_value, max_value)


def plot_all_1d_ale_horizons(data, output_folder):
    
    features = list(data[0]['ale1d'].keys())
    N_train = data[0]['train_lengths']
    
    for feature in features:
        ales = [d['ale1d'][feature] for d in data]
        horizons = [d['period'] for d in data]
        plot_horizons_1d_ale(feature, ales, horizons, N_train, output_folder)
    


def plot_all_2d_ale(ale_dict, output_folder):
    for features, ale2d in ale_dict['ale2d'].items():
        first_feature, second_feature = features.split(':')
        ale1d_first = ale_dict['ale1d'][first_feature]
        ale1d_second = ale_dict['ale1d'][second_feature]
        
        plot_single_1d_2d_ale(first_feature, second_feature, 
                              ale1d_first, ale1d_second,
                              ale2d, output_folder)
        
        plot_single_2d_ale(first_feature, second_feature, 
                           ale2d, output_folder)
        
        print(features, ':', ale2d['mean_ale'].abs().sum())
    


if __name__ == '__main__':
    jobs = [f for f in os.listdir(FINAL_ALE_RUN) if f.startswith('job')]
    jobs.sort(key = lambda s: int(s.replace('job-', '')))
    
    data = []
    
    for job in jobs:
        with open(FINAL_ALE_RUN + '%s/ale.pickle' % job, "rb") as f:
            ale_1d = pickle.load(f)
        
        results = pd.read_hdf(FINAL_ALE_RUN + '%s/model_results.hdf' % job)
        
        job_info = extract_job_info(results)
        
        data.append({
                'ale1d': ale_1d,
                'train_lengths': list(map(lambda sr: sr['train_length'], results['split_results'])),
                **job_info
                })
        
        min_1d_ale = min(list(map(lambda ale: ale.mean_ale.min(), ale_1d.values())))
        max_1d_ale = max(list(map(lambda ale: ale.mean_ale.max(), ale_1d.values())))
        
        cut = max(abs(min_1d_ale), abs(max_1d_ale))
        cut = np.round(cut + 0.1, 1)
        
        output_folder = FINAL_ALE_RUN + '%s/ale/' % job
        os.makedirs(output_folder, exist_ok = True)
        
        plot_all_1d_ale(data[-1], output_folder)
        
        output_folder = FINAL_ALE_RUN + '%s/ale-same_scale/' % job
        os.makedirs(output_folder, exist_ok = True)
        
        plot_all_1d_ale(data[-1], output_folder, min_value = -cut, max_value = cut)
        #break
    
    output_folder = FINAL_ALE_RUN + '/ale-horizons/'
    os.makedirs(output_folder, exist_ok = True)
    
    plot_all_1d_ale_horizons(data, output_folder)
    
    
    output_folder = FINAL_ALE_RUN + '/paper-ale/'
    os.makedirs(output_folder, exist_ok = True)
    
    plot_earnings_sales_1d_ale(data, output_folder)
    plot_call_related_1d_ale(data, output_folder)
    plot_valuation_related_1d_ale(data, output_folder)
    
    
    jobs = [f for f in os.listdir(FINAL_ALE_2D_RUN) if f.startswith('job')]
    jobs.sort(key = lambda s: int(s.replace('job-', '')))
    
    data = []
    
    for job in jobs:
        with open(FINAL_ALE_RUN + '%s/ale.pickle' % job, "rb") as f:
            ale_1d = pickle.load(f)
        
        with open(FINAL_ALE_2D_RUN + '%s/ale_2d.pickle' % job, "rb") as f:
            ale_2d = pickle.load(f)
        
        results = pd.read_hdf(FINAL_ALE_2D_RUN + '%s/model_results.hdf' % job)
        
        job_info = extract_job_info(results)
        
        data.append({
                'ale1d': ale_1d,
                'ale2d': ale_2d,
                'train_lengths': list(map(lambda sr: sr['train_length'], results['split_results'])),
                **job_info
                })
        
        output_folder = FINAL_ALE_2D_RUN + '%s/ale/' % job
        os.makedirs(output_folder, exist_ok = True)
        
        plot_all_2d_ale(data[-1], output_folder)
        
        break
    

