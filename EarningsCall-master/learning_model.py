#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:39:45 2019

@author: mschnaubelt
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.base import is_regressor
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
from sklearn.metrics import explained_variance_score, mean_squared_error
import statsmodels.formula.api as smf
import scipy.stats as sst
from scipy.stats import mannwhitneyu, ttest_ind
import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence

from analysis.feature_dependence import calculate_partial_dependeces, \
                                        calculate_accumulated_local_effects, \
                                        calculate_2d_accumulated_local_effects
from analysis.feature_importance import calculate_permutation_importances

from config import BASE_FEATURES


def directional_bacc(y_true, y_pred):
    
    return balanced_accuracy_score((y_true >= 0.0)*1.0, (y_pred >= 0.0)*1.0)


def aggregate_predictions(results, use_signed_returns = False):
    
    col = 'signed_target_return' if use_signed_returns else 'return_target'
    
    reg = smf.ols(col + ' ~ 1', data = results).fit(cov_type = 'HAC', 
                 cov_kwds = {'maxlags': 10})
    t = reg.params[0] / reg.HC0_se[0]
    
    agg_values = {
                    'frac': results.target.mean(),
                    't': t,
                    'std_err': reg.HC0_se[0]
                 }
    
    agg_values.update(results[col].describe())
    
    if use_signed_returns: # compute also median test between flop and top preds
        r_top = results.loc[results.pred == 1]
        r_flop = results.loc[results.pred == 0]
        
        mannwhitneyu_result = mannwhitneyu(r_top.signed_target_return, r_flop.signed_target_return, 
                                           alternative = 'greater')
        
        agg_values['median_pvalue'] = mannwhitneyu_result.pvalue
    
    return pd.Series(agg_values)



def calculate_job_result(results, metrics, alpha):
    job_results = {}
    
    job_results.update(aggregate_predictions(results, 
                                             use_signed_returns = True).to_dict())
    
    for m_name, metric in metrics.items():
        job_results['test_' + m_name] = metric(results.target, results.pred)
    job_results['test_return_ge_0'] = (results.signed_target_return>=0).mean()
    
    deciles = pd.cut(results.pred_rank, np.arange(0.0, 1.01, 0.1))
    deciles_rets = results.groupby(deciles).apply(aggregate_predictions)
    job_results['deciles_returns'] = deciles_rets
    
    top_flop_results = results.loc[(results.pred_rank > 1.0 - alpha) | (results.pred_rank < alpha)]
    tfd = aggregate_predictions(top_flop_results, use_signed_returns = True).to_dict()
    job_results.update({'top_flop_' + k: v for k, v in tfd.items()})
    
    for m_name, metric in metrics.items():
        job_results['top_flop_' + m_name] = metric(top_flop_results.target, 
                                                   top_flop_results.pred)
    job_results['top_flop_return_ge_0'] = (top_flop_results.signed_target_return>=0).mean()
    
    
    daily_returns = top_flop_results.groupby('local_date').signed_target_return.mean()
    job_results.update({'daily_top_flop_' + k: v for k, v in daily_returns.describe().to_dict().items()})
    
    reg = smf.ols('signed_target_return ~ 1', data = pd.DataFrame(daily_returns)).fit(
            cov_type = 'HAC', cov_kwds = {'maxlags': 10})
    t = reg.params[0] / reg.HC0_se[0]
    
    job_results['daily_top_flop_t'] = t
    
    return job_results




def run_model(all_data, job, job_dir = None):
    index_names = ['SP500TR', 'SP400TR', 'SP600TR'] if job['train_subset'] == 'SP1500' \
                    else [job['train_subset']]
    if job['train_subset'] is not None:
        data = all_data.loc[all_data.mkt_index.isin(index_names)].copy()
    else:
        data = all_data
    
    data = data.sort_values('final_datetime')
    
    
    model_base = job['model']
    
    is_regr = is_regressor(model_base)
    
    returns = all_data[job['train_target']]
    train_target = returns if is_regr else (returns > 0.0)*1.0
    return_target = all_data[job['return_target']]
    
    feature_data = all_data[job['features']]
    
    
    all_results = []
    split_results = []
    
    trained_models = []
    trained_X_data = []
    trained_y_return = []
    
    if not is_regr:
        metrics = {'mcc': matthews_corrcoef, 'bacc': balanced_accuracy_score}
    else:
        metrics = {'r2': explained_variance_score, 'mse': mean_squared_error,
                   'bacc': directional_bacc}
    
    
    val = job['validator']
    
    for i, (train_index, test_index, val_index) in enumerate(val.split(data)):
        
        split_result = {}
        
        print("Split #%d" % i)
        print("Train data of length %d from %s to %s" % (len(data.loc[train_index]), 
                            data.loc[train_index].final_datetime.min(), 
                            data.loc[train_index].final_datetime.max()))
        print("Validation data of length %d from %s to %s" % (len(data.loc[val_index]), 
                            data.loc[val_index].final_datetime.min(), 
                            data.loc[val_index].final_datetime.max()))
        print("Test data of length %d from %s to %s" % (len(data.loc[test_index]), 
                            data.loc[test_index].final_datetime.min(), 
                            data.loc[test_index].final_datetime.max()))
        
        model = sklearn.base.clone(model_base)
        
        
        abs_ret = all_data.loc[train_index, job['train_target']].abs()
        
        w = 1 + abs_ret / abs_ret.median()
        w = w / w.mean()
        
        add_params = {}
        
        train_X = feature_data.loc[train_index]
        
        if not is_regr:
            if 'Pipeline' in model.__class__.__name__ and model.steps[1][0] == 'ANN':
                #add_params['ANN__sample_weight'] = w.loc[train_index].values
                
                import keras
                earl = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 25, min_delta = 0)
                
                add_params['ANN__callbacks'] = [earl]
            elif 'Pipeline' in model.__class__.__name__:
                #add_params['LR__sample_weight'] = w.loc[train_index]
                
                wd = pd.DataFrame(sst.mstats.winsorize(feature_data.loc[train_index], limits = (0.01, 0.01)),
                                                       index = feature_data.loc[train_index].index,
                                                       columns = feature_data.loc[train_index].columns)
                train_X = wd
            elif 'LogisticRegression' in model.__class__.__name__:
                #add_params['sample_weight'] = w.loc[train_index]
                
                wd = pd.DataFrame(sst.mstats.winsorize(feature_data.loc[train_index], limits = (0.01, 0.01)),
                                                       index = feature_data.loc[train_index].index,
                                                       columns = feature_data.loc[train_index].columns)
                train_X = wd
            else:
                pass
                #add_params['sample_weight'] = w.loc[train_index]
        else:
            if 'Regression' in model.__class__.__name__ or 'Lasso' in str(model):
                wd = pd.DataFrame(sst.mstats.winsorize(feature_data.loc[train_index], limits = (0.01, 0.01)),
                                                       index = feature_data.loc[train_index].index,
                                                       columns = feature_data.loc[train_index].columns)
                train_X = wd
        
        model.fit(train_X, 
                  train_target.loc[train_index], 
                  **add_params
                  )
        
        trained_models.append(model)
        trained_X_data.append(feature_data.loc[train_index])
        trained_y_return.append(return_target.loc[train_index])
        
        
        union_index = train_index.union(val_index).union(test_index)
        val_test_index = val_index.union(test_index)
        
        preds = pd.Series(model.predict(feature_data.loc[union_index]), 
                              index = union_index)
        
        if is_regr:
            pred_ps = pd.concat([preds, preds], axis = 1)
        else:
            pred_ps = pd.DataFrame(model.predict_proba(feature_data.loc[union_index]), 
                                   index = union_index)
        
        
        val_data = [('train_', train_target.loc[train_index], preds.loc[train_index]), 
             ('test_', train_target.loc[val_test_index], preds.loc[val_test_index]),
             ('val_test_', train_target.loc[test_index], preds.loc[test_index])]
        
        for prefix, y_true, y_pred in val_data:
            for m_name, metric in metrics.items():
                m = metric(y_pred, y_true)
                split_result[prefix + m_name] = m
                
                split_result[prefix + 'length'] = len(y_true)
        
        
        #ranks = pred_ps.loc[val_test_index, 1].rolling(window = job['rolling_window_size']).\
        #                apply(lambda x: x.rank(pct = True).iloc[-1], raw = False)
        
        sorted_val_test_index = data.loc[val_test_index, 'final_datetime'].sort_values().index
        
        ranks = pred_ps.loc[sorted_val_test_index, 1].rolling(window = job['rolling_window_size']).\
                        apply(lambda x: pd.DataFrame(x).rank(pct = True).iloc[-1])
        
        
        if is_regr:
            signed_returns = ((2*(ranks.loc[test_index]>0.5) -1) * return_target.loc[test_index])
        else:
            signed_returns = ((2*preds.loc[test_index] -1) * return_target.loc[test_index])
        
        if not is_regr:
            print(split_result['train_mcc'], split_result['test_mcc'], signed_returns.mean()*1E4)
            
            split_result['val_test_return'] = signed_returns.mean()
        
        
        result = pd.concat([pd.Series(i, index = test_index),
                            train_target.loc[test_index],
                            preds.loc[test_index], 
                            pred_ps.loc[test_index, 1],
                            ranks.loc[test_index],
                            return_target.loc[test_index],
                            signed_returns], 
                            axis = 1)
        
        result.columns = ['split', 'target', 'pred', 'pred_p', 'pred_rank', 
                          'return_target', 'signed_target_return']
        
        result = pd.concat([data.loc[test_index, ['filename', 'file_id', 
                                                  'local_date', 'ticker_symbol', 
                                                  'fiscal_period', 'mkt_index',
                                                  'final_datetime', 'release_datetime',
                                                  'naics', 'naics_sec', 'naics_subsec', 
                                                  'trbc', 'trbc_sec',
                                                  'EPS Report Date'] + BASE_FEATURES],
                            result],
                            axis = 1)
        
        
        if job['calculate_permutation_feature_importances']:
            pfi = calculate_permutation_importances(model, 
                                                    X = feature_data.loc[test_index], 
                                                    y = train_target.loc[test_index], 
                                                    y_return = return_target.loc[test_index],
                                                    y_ranks = ranks,
                                                    top_flop_cutoff = job['top_flop_cutoff'])
            split_result.update(pfi)
            
        
        if 'feature_importances_' in dir(model):
            split_result['feature_imp_gini'] = dict(zip(feature_data.columns.tolist(), 
                        model.feature_importances_))
        
        
            
        all_results.append(result)
        split_results.append(split_result)
        
        #break
    
    results = pd.concat(all_results)
    
    
    if 'calculate_partial_dependence' in job and job['calculate_partial_dependence']:
        print("Calculating single partial dependence")
        
        features = [[f] for f in list(feature_data.columns)]
        
        calculate_partial_dependeces(trained_models, trained_X_data, 
                                     trained_y_return, features,
                                     out_dir = job_dir)
        
    
    if 'calculate_dual_partial_dependence' in job and job['calculate_dual_partial_dependence']:
        print("Calculating dual partial dependence")
        fig = plt.figure(figsize=(10, 10))
        plot_partial_dependence(model, feature_data.loc[train_index], 
                                [('earnings_surprise', 'earnings_ratio'),
                                 ('earnings_surprise', 'earnings_surprise_std')], 
                                feature_names = list(feature_data.loc[train_index].columns),
                                grid_resolution = 100, percentiles = (0.02, 0.98),
                                n_cols = 2, fig = fig, n_jobs=-1)
        fig.tight_layout()
        if job_dir:
            fig.savefig(job_dir + '/dual_partial_dependence.pdf')
        
    
    if 'calculate_single_ale' in job and job['calculate_single_ale']:
        print("Calculating single ALE")
        
        calculate_accumulated_local_effects(trained_models, trained_X_data, 
                                            trained_y_return, feature_data.columns,
                                            multiprocessing = '-5' not in job['train_target'],
                                            out_dir = job_dir)
        
    if 'calculate_dual_ale' in job:
        print("Calculating dual ALE")
        
        feature_sets = job['calculate_dual_ale']
        calculate_2d_accumulated_local_effects(trained_models, trained_X_data, 
                                               trained_y_return, feature_sets, 
                                               out_dir = job_dir)
    
    
    
    job_results = calculate_job_result(results, metrics, job['top_flop_cutoff'])
    
    job_results['split_results'] = split_results
    
    feature_imp_perm = pd.DataFrame([sr['feature_imp_perm'] for sr in split_results
                                     if 'feature_imp_perm' in sr])
    feature_imp_gini = pd.DataFrame([sr['feature_imp_gini'] for sr in split_results 
                                     if 'feature_imp_gini' in sr])
    
    for mkt in results.mkt_index.unique():
        mkt_results = results[results.mkt_index == mkt]
        job_results[mkt + '_results'] = calculate_job_result(mkt_results, metrics, job['top_flop_cutoff'])
    
    mkt_results = results[results.mkt_index !='NONE:US']
    job_results['SP1500_results'] = calculate_job_result(mkt_results, metrics, job['top_flop_cutoff'])
    
    job_results['feature_imp_perm'] = feature_imp_perm
    job_results['feature_imp_gini'] = feature_imp_gini
    
    job_results['job'] = job
    
    
    return results, job_results



def summarize_model_results(job, r):
    job_summary = pd.Series(job)
    
    job_summary['validator'] = str(job_summary['validator'])
    job_summary['model'] = str(job_summary['model'])
    
    m_names = [c.replace('test_', '') for c in r if c in ['test_mcc', 'test_bacc', 'test_r2', 'test_mse']]
    
    r_summary = {
            'test.return.mean': r['mean']*1E4,
            'test.return.median': r['50%']*1E4,
            'test.top_flop.count': r['top_flop_count'],
            'test.top_flop.abn_ret.mean': r['top_flop_mean']*1E4,
            'test.top_flop.abn_ret.mean_t': r['top_flop_t'],
            'test.top_flop.abn_ret.median': r['top_flop_50%']*1E4,
            'test.top_flop.daily_ret.mean': r['daily_top_flop_mean']*1E4,
            'test.top_flop.daily_ret.mean_t': r['daily_top_flop_t']
            }
    
    for m_name in m_names:
        r_summary['test.%s' % m_name] = r['test_' + m_name]
        r_summary['test.top_flop.%s' % m_name] = r['top_flop_' + m_name]
    
    subsets = [('NONE:US', 'non_sp'), ('SP1500', 'sp1500'), 
               ('SP500TR', 'sp500'), ('SP400TR', 'sp400'), ('SP600TR', 'sp600')]
    
    for mkt_index, n in subsets:
        if (mkt_index + '_results') in r:
            r_summary.update({
                n + '.top_flop.count': r[mkt_index + '_results']['top_flop_count'],
                #n + '.top_flop.mcc': r[mkt_index + '_results']['top_flop_mcc'],
                n + '.top_flop.mean': r[mkt_index + '_results']['top_flop_mean']*1E4,
                n + '.top_flop.mean_t': r[mkt_index + '_results']['top_flop_t'],
                n + '.top_flop.daily_ret.mean': r[mkt_index + '_results']['daily_top_flop_mean']*1E4
            })
    
    r_summary = pd.Series(r_summary)
    
    return pd.concat([job_summary, r_summary])


