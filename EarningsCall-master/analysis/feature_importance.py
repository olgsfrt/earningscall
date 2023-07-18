#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:19:57 2019

@author: mschnaubelt
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import matthews_corrcoef, make_scorer, get_scorer
from sklearn.base import is_regressor

from eli5.sklearn import PermutationImportance

from analysis.analysis_helper import read_run
from analysis.return_score import return_score
from config import FINAL_ALE_RUN, FEATURE_NAME_DICT




def calculate_permutation_importances(model, X, y, y_return, y_ranks, top_flop_cutoff):
    
    if is_regressor(model):
        scorer = get_scorer('neg_mean_squared_error')
    else:
        scorer = make_scorer(matthews_corrcoef, greater_is_better = True)
    
    result = {}
    
    perm = PermutationImportance(model, random_state = 1, n_iter = 10, 
                                 scoring = scorer).fit(X, y)
    
    result['feature_imp_perm'] = dict(zip(X.columns.tolist(), 
                perm.feature_importances_))
    
    result['feature_imp_perm_std'] = dict(zip(X.columns.tolist(), 
                perm.feature_importances_std_))
    
    if not is_regressor(model):
        scorer = make_scorer(return_score, greater_is_better = True, 
                             abnormal_return = y_return)
        
        perm = PermutationImportance(model, random_state = 1, n_iter = 5, 
                                     scoring = scorer).fit(X, y)
        
        result['feature_return_imp_perm'] = dict(zip(X.columns.tolist(), 
                    perm.feature_importances_))
        
        result['feature_return_imp_perm_std'] = dict(zip(X.columns.tolist(), 
                    perm.feature_importances_std_))
    
    
#    rank_index = ((y_ranks > (1 - top_flop_cutoff)) | (y_ranks < top_flop_cutoff))
#    rank_index = rank_index[rank_index].index
#    
#    
#    scorer = make_scorer(matthews_corrcoef, greater_is_better = True)
#    
#    perm = PermutationImportance(model, random_state = 1, n_iter = 5, 
#                                 scoring = scorer).fit(X.loc[rank_index], y.loc[rank_index])
#    
#    result['feature_imp_perm_top_flop'] = dict(zip(X.columns.tolist(), 
#                perm.feature_importances_))
#    
#    
#    scorer = make_scorer(return_score, greater_is_better = True, 
#                         abnormal_return = y_return.loc[rank_index])
#    
#    perm = PermutationImportance(model, random_state = 1, n_iter = 5, 
#                                 scoring = scorer).fit(
#        X.loc[rank_index], y.loc[rank_index])
#    
#    result['feature_return_imp_perm_top_flop'] = dict(zip(X.columns.tolist(), 
#                perm.feature_importances_))
    
    return result


def analyze_feature_importance(result, output_basefilename = None):
    imps = []
    
    for imp_col in ['feature_imp_gini', 'feature_imp_perm', 'feature_return_imp_perm', 'feature_imp_perm_top_flop']:
        if not all([imp_col in r for r in result['split_results']]):
            continue
        
        f_imp = pd.DataFrame([r[imp_col] for r in result['split_results']])
        Ls = pd.Series([r['test_length'] for r in result['split_results']])
        
        imps.append(pd.DataFrame(f_imp.mul(Ls, axis=0).sum() / Ls.sum(), 
                                 columns = [imp_col]))
    
    
    imps = pd.concat(imps + [pd.DataFrame()], axis = 1)
    
    if output_basefilename:
        imps.to_csv(output_basefilename + '.csv')
    
    return imps



def plot_feature_importance(ax, imps, n_features = 15, x_max = None):
    plt_imps = imps.sort_values(ascending = False).head(n_features).sort_values()
    
    ax.barh(range(n_features), plt_imps, )
    
    ax.set_yticks([])
    ax.set_yticklabels(['']*n_features)
    
    for i in range(n_features):
        text = FEATURE_NAME_DICT[plt_imps.index[i]] if plt_imps.index[i] \
                    in FEATURE_NAME_DICT else plt_imps.index[i]
        ax.annotate(text, 
                    (plt_imps.iloc[i], i), 
                    xytext = (5, -1.5),
                    textcoords = 'offset points',
                    fontsize = 11, ha = "left", va = 'center')
        
        text = '%.1f' % (plt_imps.iloc[i])
        ax.annotate(text, 
                    (plt_imps.iloc[i], i), 
                    xytext = (-3, -0.5),
                    textcoords = 'offset points',
                    color = 'white',
                    fontsize = 9, ha = "right", va = 'center')
    
    ax.set_ylim(-0.5, n_features - 0.5)
    
    if x_max is None:
        _, upper_lim = ax.get_xlim()
        ax.set_xlim(0, upper_lim * 1.1)
    else:
        ax.set_xlim(0, x_max)



def analyze_feature_importances(run_folder):
    results = read_run(run_folder)
    R = []
    for (model, period, feat), row in results.iterrows():
        FI = analyze_feature_importance(row)
        
        FI['feature_imp_perm'] -= FI['feature_imp_perm'].min()
        FI['feature_imp_perm'] /= FI['feature_imp_perm'].sum()
        
        FI = pd.concat([FI], keys = [model], names = ['model'])
        FI = pd.concat([FI], keys = [period], names = ['period'])
        
        R.append(FI)
    
    R = pd.concat(R).unstack(level = [0, 1])
    R = R.swaplevel(0, 1, axis = 1).swaplevel(1, 2, axis = 1).swaplevel(0, 1, axis = 1)
    R.sort_index(axis = 1, inplace = True)
    
    R *= 100
    R = R.loc[R.sum(axis = 1).sort_values(ascending = False).index]
    
    R.to_excel(run_folder + 'all_feature_imps.xlsx')
    
    ni  = R.index.map(lambda c: FEATURE_NAME_DICT[c] if c in FEATURE_NAME_DICT else c)
    R.set_index(ni, inplace = True)
    L = R.to_latex(float_format = '%.2f', multicolumn = True, escape = False)
    with open(run_folder + '/all_feature_imps.tex', "w") as f:
        print(L, file = f)
    
    
    fig = plt.figure(figsize = (10, 12))
    axs = fig.subplots(4, 2)
    
    for col, t in enumerate(['feature_imp_gini', 'feature_imp_perm']):
        for row, p in enumerate([-5, 5, 20, 60]):
            imps = R.loc[:, ('8 RF-D20-E5000', p, t)]
            ax = axs[row, col]
            
            plot_feature_importance(ax, imps)
            
            ax.annotate('$T=%d$' % p, 
                        (0.95, 0.05),
                        xycoords = 'axes fraction',
                        fontsize = 12, 
                        ha = "right", va = 'center')
            
            ax.get_xaxis().set_visible(False)
            
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #ax.axis('off')
    

    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.1, wspace = 0.1,
                        left = 0.02, right = 0.98,
                        bottom = 0.01, top = 0.99)
    
    fig.savefig(run_folder + 'feature_imp_matrix.pdf')
    plt.close()
    
    
    fig = plt.figure(figsize = (10, 4))
    axs = fig.subplots(1, 3)
    
    for col, p in enumerate([-5, 5, 60]):
        imps = R.loc[:, ('8 RF-D20-E5000', p, 'feature_imp_gini')]
        ax = axs[col]
        
        plot_feature_importance(ax, imps, x_max = 18)
        
        ax.annotate('$T=%d$' % p, 
                    (0.98, 0.02),
                    xycoords = 'axes fraction',
                    fontsize = 12, 
                    ha = "right", va = 'center')
        
        ax.get_xaxis().set_visible(False)
        
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.1, wspace = 0.1,
                        left = 0.02, right = 0.98,
                        bottom = 0.01, top = 0.99)
    
    fig.savefig(run_folder + 'feature_imp_gini.pdf')
    plt.close()
    
    
    
    fig = plt.figure(figsize = (10, 4))
    axs = fig.subplots(1, 3)
    
    for col, p in enumerate([-5, 5, 60]):
        imps = R.loc[:, ('8 RF-D20-E5000', p, 'feature_imp_perm')]
        ax = axs[col]
        
        plot_feature_importance(ax, imps)#, x_max = 60)
        
        ax.annotate('$T=%d$' % p, 
                    (0.98, 0.02),
                    xycoords = 'axes fraction',
                    fontsize = 12, 
                    ha = "right", va = 'center')
        
        ax.get_xaxis().set_visible(False)
        
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.1, wspace = 0.1,
                        left = 0.02, right = 0.98,
                        bottom = 0.01, top = 0.99)
    
    fig.savefig(run_folder + 'feature_imp_perm.pdf')
    plt.close()
    
    
    #    for job in jobs_config:
#        job_results = results.loc[job['index']]
#        ax = axs[job['row'], job['column']]
#        
#        y_min, y_max = plot_deciles(job_results, ax)
#        
#        global_y_min, global_y_max = min(global_y_min, y_min), max(global_y_max, y_max)
#    
#    #global_y_min = - global_y_max
#    
#    for job in jobs_config:
#        ax = axs[job['row'], job['column']]
#        ax.set_ylim(global_y_min - 65, global_y_max + 65)
#        
#        if job['column'] == cols - 1:
#            ax.yaxis.set_label_position("right")
#            ax.set_ylabel('Mean abnormal return', fontsize = 12)
#        
#        if job['row'] == rows - 1:
#            ax.set_xlabel('decile', fontsize = 12)
#            ax.set_xticklabels(range(1, 11))
#        
#        if job['row'] == 0:
#            ax.annotate('%s' % job['model'],
#                        xy = (0.5, 1.0), xycoords = 'axes fraction',
#                        size = 13, ha = 'center', va = 'bottom')
#        
#        if job['column'] == 0:
#            ax.annotate('%d' % job['period'],
#                        xy = (-0.01, 0.5), xycoords = 'axes fraction',
#                        size = 13, ha = 'right', va = 'center')
#        
#        ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
#    #plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")
    
    


if __name__ == '__main__':
    analyze_feature_importances(run_folder = FINAL_ALE_RUN)

    

    


