#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:16:18 2019

@author: mschnaubelt
"""

import pickle
import logging
import pandas as pd
import numpy as np
import multiprocessing as mp

import matplotlib.pyplot as plt
import seaborn as sns

from analysis.pdp import partial_dependence
from analysis.ale import _first_order_ale_quant, _second_order_ale_quant

#Original function

""""
def calculate_partial_dependeces(models, Xs, y_returns, features, out_dir = None):
    
    result = {}
    
    with mp.Pool(int(mp.cpu_count()/2), maxtasksperchild = 1) as pool:
        for feature in features:
            pdps = []
            
            joined_X = pd.concat([X[feature] for X in Xs])
            grid = joined_X.quantile(np.linspace(0, 1, 51)).values
            
            for i, model in enumerate(models):
                logging.info("Computing for feature %s and model %d", feature, i)
                X = Xs[i]
                y_return = y_returns[i]
                model.n_jobs = 1
                
                pdps.append(partial_dependence(model, X, y_return, feature, grid, pool))
            
            result[':'.join(feature)] = pdps
    
    if out_dir:
        with open(out_dir + '/pdep.pickle', "wb") as f:
            pickle.dump(result, f)
        
        for key, pdps in result.items():
            J = pd.concat([r.set_index('val').mean_pred for r in pdps], axis = 1)
            K = pd.concat([r.set_index('val').mean_abs_pred for r in pdps], axis = 1)
            
            fig = plt.figure(figsize = (8, 8))
            ax1, ax2 = fig.subplots(2, 1)
            
            for ci in range(len(J.columns)):
                ax1.plot(J.iloc[2:-2, ci], label = '%d' % ci)
            
            ax1.plot(J.iloc[2:-2].mean(axis = 1), 'k', label = 'mean')
            ax2.plot(K.iloc[2:-2].mean(axis = 1), 'k', label = 'mean')
            
            ax1.legend()
            
            fig.tight_layout()
            fig.savefig('%s/pdep_%s.pdf' % (out_dir, key.replace('/', '_')))
            plt.close()
    
    return result

def calculate_partial_dependences(models, Xs, y_returns, features, out_dir=None):
    result = {}
    if __name__ == '__main__':
        with mp.Pool(int(mp.cpu_count()/2), maxtasksperchild=1) as pool:
            for feature in features:
                pdps = []
                
                joined_X = pd.concat([X[feature] for X in Xs])
                grid = joined_X.quantile(np.linspace(0, 1, 51)).values
                
                for i, model in enumerate(models):
                    logging.info("Computing for feature %s and model %d", feature, i)
                    X = Xs[i]
                    y_return = y_returns[i]
                    model.n_jobs = 1
                    
                    pdps.append(partial_dependence(model, X, y_return, feature, grid, pool))
                
                result[':'.join(feature)] = pdps
        
        if out_dir:
            with open(out_dir + '/pdep.pickle', "wb") as f:
                pickle.dump(result, f)
            
            for key, pdps in result.items():
                J = pd.concat([r.set_index('val').mean_pred for r in pdps], axis = 1)
                K = pd.concat([r.set_index('val').mean_abs_pred for r in pdps], axis = 1)
                
                fig = plt.figure(figsize = (8, 8))
                ax1, ax2 = fig.subplots(2, 1)
                
                for ci in range(len(J.columns)):
                    ax1.plot(J.iloc[2:-2, ci], label = '%d' % ci)
                
                ax1.plot(J.iloc[2:-2].mean(axis = 1), 'k', label = 'mean')
                ax2.plot(K.iloc[2:-2].mean(axis = 1), 'k', label = 'mean')
                
                ax1.legend()
                
                fig.tight_layout()
                fig.savefig('%s/pdep_%s.pdf' % (out_dir, key.replace('/', '_')))
                plt.close()
        
        return result
"""
def calculate_partial_dependeces(models, Xs, y_returns, features, out_dir = None):
    result = {}
    for feature in features:
        pdps = []
        joined_X = pd.concat([X[feature] for X in Xs])
        grid = joined_X.quantile(np.linspace(0, 1, 51)).values
        for i, model in enumerate(models):
            logging.info("Computing for feature %s and model %d", feature, i)
            X = Xs[i]
            y_return = y_returns[i]
            model.n_jobs = 1
            pdps.append(partial_dependence(model, X, y_return, feature, grid))
        result[':'.join(feature)] = pdps

    if out_dir:
        with open(out_dir + '/pdep.pickle', "wb") as f:
            pickle.dump(result, f)
        for key, pdps in result.items():
            J = pd.concat([r.set_index('val').mean_pred for r in pdps], axis = 1)
            K = pd.concat([r.set_index('val').mean_abs_pred for r in pdps], axis = 1)
            fig = plt.figure(figsize = (8, 8))
            ax1, ax2 = fig.subplots(2, 1)
            for ci in range(len(J.columns)):
                ax1.plot(J.iloc[2:-2, ci], label = '%d' % ci)
            ax1.plot(J.iloc[2:-2].mean(axis = 1), 'k', label = 'mean')
            ax2.plot(K.iloc[2:-2].mean(axis = 1), 'k', label = 'mean')
            ax1.legend()
            fig.tight_layout()
            fig.savefig('%s/pdep_%s.pdf' % (out_dir, key.replace('/', '_')))
            plt.close()
    return result


def _ale_worker(X, model, i, feature, grid):
    print('Computing ALE for model %d and feature %s' % (i, feature))
    model.n_jobs = 1
    return _first_order_ale_quant(model.predict, X, feature, grid)

"""
#Original function
def calculate_accumulated_local_effects(models, Xs, y_returns, features, 
                                        out_dir = None, multiprocessing = True):
    
    result = {}
    
    with mp.Pool(len(models) if multiprocessing else 1, maxtasksperchild = 1) as pool:
        for feature in features:
            
            joined_X = pd.concat([X[feature] for X in Xs])
            grid = joined_X.quantile(np.linspace(0, 1, 51)).values
            
            jobs = [(Xs[i], models[i], i, feature, grid) for i in range(len(models))]
            
            if multiprocessing:
                ales = pool.starmap(_ale_worker, jobs)
            else:
                ales = list(map(lambda j: _ale_worker(j[0], j[1], j[2], j[3], j[4]), jobs))
            
            ales = pd.DataFrame(ales).transpose()
            ales['mean_ale'] = ales.mean(axis = 1)
            ales['std_ale'] = ales.std(axis = 1)
            ales['lower_value'] = grid[:-1]
            ales['upper_value'] = grid[1:]
            
            result[feature] = ales
    
    if out_dir:
        with open(out_dir + '/ale.pickle', "wb") as f:
            pickle.dump(result, f)
        
        for key, ales in result.items():
            fig = plt.figure(figsize = (8, 5))
            ax1 = fig.subplots(1, 1)
            
            x_values = (ales['lower_value'] + ales['upper_value']) / 2
            
            for ci in [c for c in ales.columns if type(c) == int]:
                ax1.plot(x_values.iloc[2:-2], ales.iloc[2:-2][ci], label = '%d' % ci)
            
            ax1.plot(x_values.iloc[2:-2], ales.iloc[2:-2]['mean_ale'], 'k', label = 'mean')
            
            ax1.legend()
            
            fig.tight_layout()
            fig.savefig('%s/ale_%s.pdf' % (out_dir, key.replace('/', '_')))
            plt.close()
    
    
    return result


def calculate_accumulated_local_effects(models, Xs, y_returns, features, out_dir=None, multiprocessing=True):
    result = {}
    if __name__ == '__main__':
        with mp.Pool(len(models) if multiprocessing else 1, maxtasksperchild=1) as pool:
            
            for feature in features:
                
                joined_X = pd.concat([X[feature] for X in Xs])
                grid = joined_X.quantile(np.linspace(0, 1, 51)).values
                
                jobs = [(Xs[i], models[i], i, feature, grid) for i in range(len(models))]
                
                if multiprocessing:
                    ales = pool.starmap(_ale_worker, jobs)
                else:
                    ales = list(map(lambda j: _ale_worker(j[0], j[1], j[2], j[3], j[4]), jobs))
                
                ales = pd.DataFrame(ales).transpose()
                ales['mean_ale'] = ales.mean(axis = 1)
                ales['std_ale'] = ales.std(axis = 1)
                ales['lower_value'] = grid[:-1]
                ales['upper_value'] = grid[1:]
                
                result[feature] = ales
        
        if out_dir:
            with open(out_dir + '/ale.pickle', "wb") as f:
                pickle.dump(result, f)
            
            for key, ales in result.items():
                fig = plt.figure(figsize = (8, 5))
                ax1 = fig.subplots(1, 1)
                
                x_values = (ales['lower_value'] + ales['upper_value']) / 2
                
                for ci in [c for c in ales.columns if type(c) == int]:
                    ax1.plot(x_values.iloc[2:-2], ales.iloc[2:-2][ci], label = '%d' % ci)
                
                ax1.plot(x_values.iloc[2:-2], ales.iloc[2:-2]['mean_ale'], 'k', label = 'mean')
                
                ax1.legend()
                
                fig.tight_layout()
                fig.savefig('%s/ale_%s.pdf' % (out_dir, key.replace('/', '_')))
                plt.close()
        
        
        return result
"""
def calculate_accumulated_local_effects(models, Xs, y_returns, features, out_dir = None):
    result = {}
    for feature in features:
        joined_X = pd.concat([X[feature] for X in Xs])
        grid = joined_X.quantile(np.linspace(0, 1, 51)).values
        jobs = [(Xs[i], models[i], i, feature, grid) for i in range(len(models))]
        ales = list(map(lambda j: _ale_worker(j[0], j[1], j[2], j[3], j[4]), jobs))
        ales = pd.DataFrame(ales).transpose()
        ales['mean_ale'] = ales.mean(axis = 1)
        ales['std_ale'] = ales.std(axis = 1)
        ales['lower_value'] = grid[:-1]
        ales['upper_value'] = grid[1:]
        result[feature] = ales

    if out_dir:
        with open(out_dir + '/ale.pickle', "wb") as f:
            pickle.dump(result, f)
        for key, ales in result.items():
            fig = plt.figure(figsize = (8, 5))
            ax1 = fig.subplots(1, 1)
            x_values = (ales['lower_value'] + ales['upper_value']) / 2
            for ci in [c for c in ales.columns if type(c) == int]:
                ax1.plot(x_values.iloc[2:-2], ales.iloc[2:-2][ci], label = '%d' % ci)
            ax1.plot(x_values.iloc[2:-2], ales.iloc[2:-2]['mean_ale'], 'k', label = 'mean')
            ax1.legend()
            fig.tight_layout()
            fig.savefig('%s/ale_%s.pdf' % (out_dir, key.replace('/', '_')))
            plt.close()
    return result






def _2d_ale_worker(X, model, i, features, quantiles):
    print('Computing 2D ALE for model %d and features %s' % (i, features))
    model.n_jobs = 1
    return _second_order_ale_quant(model.predict, X, features, quantiles)

"""
#Original function
def calculate_2d_accumulated_local_effects(models, Xs, y_returns, feature_sets, out_dir = None):
    
    result = {}
    
    with mp.Pool(len(models), maxtasksperchild = 1) as pool:
        for feature_set in feature_sets:
            joined_X = pd.concat([X[feature_set] for X in Xs])
            quantiles = [joined_X[feature_set[0]].quantile(np.linspace(0.02, 0.98, 25)).transpose().values,
                         joined_X[feature_set[1]].quantile(np.linspace(0.02, 0.98, 25)).transpose().values]
            
            jobs = [(Xs[i], models[i], i, feature_set, quantiles) for i in range(len(models))]
            ales = pool.starmap(_2d_ale_worker, jobs)
            
            row_index = (quantiles[0][0:-1] + quantiles[0][1:]) / 2
            col_index = (quantiles[1][0:-1] + quantiles[1][1:]) / 2
            
            ales = [pd.DataFrame(ale, index=row_index, columns=col_index).stack() for ale in ales]
            for ale in ales:
                ale.index.names = feature_set
            
            ales = pd.concat(ales, axis = 1)
            ales['mean_ale'] = ales.mean(axis = 1)
            ales['std_ale'] = ales.std(axis = 1)
            
            N0 = len(quantiles[0]) - 1
            N1 = len(quantiles[1]) - 1
            ales['first_feature_lower'] = np.repeat(quantiles[0][0:-1], N1)
            ales['first_feature_upper'] = np.repeat(quantiles[0][1:], N1)
            ales['second_feature_lower'] = np.concatenate([quantiles[1][0:-1]]*N0)
            ales['second_feature_upper'] = np.concatenate([quantiles[1][1:]]*N0)
            
            result[':'.join(feature_set)] = ales
            
    
    
    if out_dir:
        with open(out_dir + '/ale_2d.pickle', "wb") as f:
            pickle.dump(result, f)
        
        for key, ales in result.items():
            fig = plt.figure(figsize = (8, 5))
            ax = fig.subplots(1, 1)
            
            mean_ale = ales['mean_ale']
            mean_ale = mean_ale.loc[~mean_ale.index.duplicated(keep='first')]
            mean_ale = mean_ale.unstack()
            #mean_ale = ales[4].unstack()
            #mean_ale = mean_ale.iloc[1:-1, 1:-1]
            
            sns.heatmap(mean_ale, 
                        cmap = 'PiYG',
                        center = 0.0,
                        xticklabels = mean_ale.columns.values.round(4),
                        yticklabels = mean_ale.index.values.round(4),
                        ax = ax)
            
            fig.tight_layout()
            fig.savefig('%s/ale_2d_%s.pdf' % (out_dir, key.replace('/', '_')))
            plt.close()
    
    
    return result


def calculate_2d_accumulated_local_effects(models, Xs, y_returns, feature_sets, out_dir=None):
    result = {}
    if __name__ == '__main__':
        with mp.Pool(len(models), maxtasksperchild = 1) as pool:
            for feature_set in feature_sets:
                joined_X = pd.concat([X[feature_set] for X in Xs])
                quantiles = [joined_X[feature_set[0]].quantile(np.linspace(0.02, 0.98, 25)).transpose().values,
                            joined_X[feature_set[1]].quantile(np.linspace(0.02, 0.98, 25)).transpose().values]
                
                jobs = [(Xs[i], models[i], i, feature_set, quantiles) for i in range(len(models))]
                ales = pool.starmap(_2d_ale_worker, jobs)
                
                row_index = (quantiles[0][0:-1] + quantiles[0][1:]) / 2
                col_index = (quantiles[1][0:-1] + quantiles[1][1:]) / 2
                
                ales = [pd.DataFrame(ale, index=row_index, columns=col_index).stack() for ale in ales]
                for ale in ales:
                    ale.index.names = feature_set
                
                ales = pd.concat(ales, axis = 1)
                ales['mean_ale'] = ales.mean(axis = 1)
                ales['std_ale'] = ales.std(axis = 1)
                
                N0 = len(quantiles[0]) - 1
                N1 = len(quantiles[1]) - 1
                ales['first_feature_lower'] = np.repeat(quantiles[0][0:-1], N1)
                ales['first_feature_upper'] = np.repeat(quantiles[0][1:], N1)
                ales['second_feature_lower'] = np.concatenate([quantiles[1][0:-1]]*N0)
                ales['second_feature_upper'] = np.concatenate([quantiles[1][1:]]*N0)
                
                result[':'.join(feature_set)] = ales
                
        
        
        if out_dir:
            with open(out_dir + '/ale_2d.pickle', "wb") as f:
                pickle.dump(result, f)
            
            for key, ales in result.items():
                fig = plt.figure(figsize = (8, 5))
                ax = fig.subplots(1, 1)
                
                mean_ale = ales['mean_ale']
                mean_ale = mean_ale.loc[~mean_ale.index.duplicated(keep='first')]
                mean_ale = mean_ale.unstack()
                #mean_ale = ales[4].unstack()
                #mean_ale = mean_ale.iloc[1:-1, 1:-1]
                
                sns.heatmap(mean_ale, 
                            cmap = 'PiYG',
                            center = 0.0,
                            xticklabels = mean_ale.columns.values.round(4),
                            yticklabels = mean_ale.index.values.round(4),
                            ax = ax)
                
                fig.tight_layout()
                fig.savefig('%s/ale_2d_%s.pdf' % (out_dir, key.replace('/', '_')))
                plt.close()
        
        
        return result
"""
def calculate_2d_accumulated_local_effects(models, Xs, y_returns, feature_sets, out_dir = None):
    result = {}
    for feature_set in feature_sets:
        joined_X = pd.concat([X[feature_set] for X in Xs])
        quantiles = [joined_X[feature_set[0]].quantile(np.linspace(0.02, 0.98, 25)).transpose().values,
                     joined_X[feature_set[1]].quantile(np.linspace(0.02, 0.98, 25)).transpose().values]
        jobs = [(Xs[i], models[i], i, feature_set, quantiles) for i in range(len(models))]
        ales = list(map(lambda j: _2d_ale_worker(j[0], j[1], j[2], j[3], j[4]), jobs))
        row_index = (quantiles[0][0:-1] + quantiles[0][1:]) / 2
        col_index = (quantiles[1][0:-1] + quantiles[1][1:]) / 2
        ales = [pd.DataFrame(ale, index=row_index, columns=col_index).stack() for ale in ales]
        for ale in ales:
            ale.index.names = feature_set
        ales = pd.concat(ales, axis = 1)
        ales['mean_ale'] = ales.mean(axis = 1)
        ales['std_ale'] = ales.std(axis = 1)
        N0 = len(quantiles[0]) - 1
        N1 = len(quantiles[1]) - 1
        ales['first_feature_lower'] = np.repeat(quantiles[0][0:-1], N1)
        ales['first_feature_upper'] = np.repeat(quantiles[0][1:], N1)
        ales['second_feature_lower'] = np.concatenate([quantiles[1][0:-1]]*N0)
        ales['second_feature_upper'] = np.concatenate([quantiles[1][1:]]*N0)
        result[':'.join(feature_set)] = ales

    if out_dir:
        with open(out_dir + '/ale_2d.pickle', "wb") as f:
            pickle.dump(result, f)
        for key, ales in result.items():
            fig = plt.figure(figsize = (8, 5))
            ax = fig.subplots(1, 1)
            mean_ale = ales['mean_ale']
            mean_ale = mean_ale.loc[~mean_ale.index.duplicated(keep='first')]
            mean_ale = mean_ale.unstack()
            sns.heatmap(mean_ale, 
                        cmap = 'PiYG',
                        center = 0.0,
                        xticklabels = mean_ale.columns.values.round(4),
                        yticklabels = mean_ale.index.values.round(4),
                        ax = ax)
            fig.tight_layout()
            fig.savefig('%s/ale_2d_%s.pdf' % (out_dir, key.replace('/', '_')))
            plt.close()
    return result




def create_ice_analysis(job, model, job_dir):
    from pycebox.ice import ice, ice_plot
    
    ice_df = ice(feature_data.loc[train_index], 'earnings_surprise', model.predict, num_grid_points=100)
    

