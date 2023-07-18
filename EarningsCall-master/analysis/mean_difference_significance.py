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

import os
import pandas as pd
import itertools

from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon, mannwhitneyu, ttest_rel
from statsmodels.stats.weightstats import ttest_ind

from dm_test import dm_test
from config import FINAL_RUN_FOLDERS
from analysis_helper import extract_job_info




def do_mcnemar(results, model1_slice, model2_slice):
    result = {}
    
    r1 = results.loc[model1_slice, 'preds']
    r2 = results.loc[model2_slice, 'preds']
    
    m1_correct = r1.target == r1.pred
    m2_correct = r2.target == r2.pred
    
    con_table = pd.crosstab(m1_correct, m2_correct)
    
#    con_table = mcnemar_table(y_target=r1.target, 
#                   y_model1=r1.pred, y_model2=r2.pred)
#    chi2, p = mcnemar(ary = con_table, corrected = True)
    
    nemar_result = mcnemar(con_table, exact = True)
    
    result['model1'] = model1_slice
    result['model2'] = model2_slice
    
    result['pvalue'] = nemar_result.pvalue
    result['statistic'] = nemar_result.statistic
    
    return result



def do_diebold_mariano(results, model1_slice, model2_slice):
    result = {}
    result['model1'] = model1_slice
    result['model2'] = model2_slice
    
    p1 = results.loc[model1_slice, 'preds']
    p2 = results.loc[model2_slice, 'preds']
    
    try:
        dm = dm_test(p1.target.tolist(), p1.pred.tolist(), p2.pred.tolist(), h = 1, crit = "MSE")
        result['pvalue'] = dm.p_value
        result['statistic'] = dm.DM
    except:
        pass
    
    return result


def do_wilcoxon_signed_rank(results, model1_slice, model2_slice):
    result = {}
    
    result['model1'] = model1_slice
    result['model2'] = model2_slice
    
    r1 = results.loc[model1_slice, 'preds']
    r2 = results.loc[model2_slice, 'preds']
    
    if model1_slice == model2_slice:
        return result
    
    wilcoxon_result = wilcoxon(r1.signed_target_return, r2.signed_target_return, 
                               zero_method = 'pratt', alternative = 'greater')
    
    result['pvalue'] = wilcoxon_result.pvalue
    result['statistic'] = wilcoxon_result.statistic
    
    return result


def do_mannwhitneyu(results, model1_slice, model2_slice, alpha = 0.1):
    result = {}
    
    result['model1'] = model1_slice
    result['model2'] = model2_slice
    
    r1 = results.loc[model1_slice, 'preds']
    r2 = results.loc[model2_slice, 'preds']
    
    r1 = r1.loc[(r1.pred_rank > 1.0 - alpha) | (r1.pred_rank < alpha)]
    r2 = r2.loc[(r2.pred_rank > 1.0 - alpha) | (r2.pred_rank < alpha)]
    
    if model1_slice == model2_slice:
        return result
    
    mannwhitneyu_result = mannwhitneyu(r1.signed_target_return, r2.signed_target_return, 
                                       alternative = 'greater')
    
    result['pvalue'] = mannwhitneyu_result.pvalue
    result['statistic'] = mannwhitneyu_result.statistic
    
    return result


def do_welcht(results, model1_slice, model2_slice, alpha = 0.1):
    result = {}
    
    result['model1'] = model1_slice
    result['model2'] = model2_slice
    
    r1 = results.loc[model1_slice, 'preds']
    r2 = results.loc[model2_slice, 'preds']
    
    r1 = r1.loc[(r1.pred_rank > 1.0 - alpha) | (r1.pred_rank < alpha)]
    r2 = r2.loc[(r2.pred_rank > 1.0 - alpha) | (r2.pred_rank < alpha)]
    
    if model1_slice == model2_slice:
        return result
    
    ttest_ind_result = ttest_ind(r1.signed_target_return, r2.signed_target_return, 
                                 alternative = 'larger', usevar = 'unequal')
    
    result['pvalue'] = ttest_ind_result[1]
    result['statistic'] = ttest_ind_result[0]
    
    return result


def do_ttest(results, model1_slice, model2_slice, alpha = 0.1):
    result = {}
    
    result['model1'] = model1_slice
    result['model2'] = model2_slice
    
    r1 = results.loc[model1_slice, 'preds']
    r2 = results.loc[model2_slice, 'preds']
    
    r1 = r1.loc[(r1.pred_rank > 1.0 - alpha) | (r1.pred_rank < alpha)]
    r2 = r2.loc[(r2.pred_rank > 1.0 - alpha) | (r2.pred_rank < alpha)]
    
    if model1_slice == model2_slice:
        return result
    
    ttest_ind_result = ttest_ind(r1.signed_target_return, r2.signed_target_return, 
                                 alternative = 'larger', usevar = 'pooled')
    
    result['pvalue'] = ttest_ind_result[1]
    result['statistic'] = ttest_ind_result[0]
    
    return result


def do_ttest_paired(results, model1_slice, model2_slice, alpha = 0.1):
    result = {}
    
    result['model1'] = model1_slice
    result['model2'] = model2_slice
    
    r1 = results.loc[model1_slice, 'preds']
    r2 = results.loc[model2_slice, 'preds']
    
    r1 = r1.loc[(r1.pred_rank > 1.0 - alpha) | (r1.pred_rank < alpha)]
    r2 = r2.loc[(r2.pred_rank > 1.0 - alpha) | (r2.pred_rank < alpha)]
    
    if model1_slice == model2_slice:
        return result
    
    ttest_rel_result = ttest_rel(r1.signed_target_return, r2.signed_target_return)
    
    result['pvalue'] = ttest_rel_result[1]
    result['statistic'] = ttest_rel_result[0]
    
    return result


def save_pvalues(rr, run_folder, file_basename):
    SR = rr.pvalue.unstack(1)
    
    SR.set_index(pd.MultiIndex.from_tuples(SR.index), inplace = True)
    SR.columns = pd.MultiIndex.from_tuples(SR.columns)
    
    writer = pd.ExcelWriter(run_folder + '/%s-pvalue.xlsx' % file_basename, 
                            engine = 'xlsxwriter')
    
    with open(run_folder + '/%s-pvalue.tex' % file_basename, "w") as f:
        L = SR.to_latex(float_format = '%.4f', multicolumn = True)
        print(L, file = f)
        SR.to_excel(writer, sheet_name = 'all')
        
        for p in [-5, 5, 20, 60]:
            SRS = SR.loc[(slice(None), p, slice(None)), (slice(None), p, slice(None))]
            L = SRS.to_latex(float_format = '%.4f', multicolumn = True)
            print(L, file = f)
            SRS.to_excel(writer, sheet_name = 'period %d' % p)
        
    writer.save()


def save_statistics(rr, run_folder, file_basename):
    SR = rr.statistic.unstack(1)
    
    SR.set_index(pd.MultiIndex.from_tuples(SR.index), inplace = True)
    SR.columns = pd.MultiIndex.from_tuples(SR.columns)
    
    writer = pd.ExcelWriter(run_folder + '/%s-stat.xlsx' % file_basename, 
                            engine = 'xlsxwriter')
    
    with open(run_folder + '/%s-stat.tex' % file_basename, "w") as f:
        L = SR.to_latex(float_format = '%.4f', multicolumn = True)
        print(L, file = f)
        SR.to_excel(writer, sheet_name = 'all')
        
        for p in [-5, 5, 20, 60]:
            SRS = SR.loc[(slice(None), p, slice(None)), (slice(None), p, slice(None))]
            L = SRS.to_latex(float_format = '%.4f', multicolumn = True)
            print(L, file = f)
            SRS.to_excel(writer, sheet_name = 'period %d' % p)
        
    writer.save()




def process_run(run_folder):
    jobs = [f for f in os.listdir(run_folder) if f.startswith('job')]
    jobs.sort(key = lambda s: int(s.replace('job-', '')))
    
    results = [pd.read_hdf(run_folder + '%s/model_results.hdf' % f) for f in jobs]
    results = pd.concat(results, axis = 1).transpose()
    
    results = results.join(results.apply(extract_job_info, axis = 1))
    
    preds = [pd.read_hdf(run_folder + '%s/predictions.hdf' % f) for f in jobs]
    results['preds'] = preds
    
    results = results.set_index(['model', 'period', 'features'])
    
    results = results[~results.index.duplicated()]
    
    base_models = list(results.loc[(slice(None), slice(None), 'FE+POL+UIQ+VR+54'), :].index.values)
    pairs = list(itertools.product(base_models, base_models))
    pairs = [p for p in pairs if (p[0][1] == p[1][1])]
    
    try:
        rr = pd.DataFrame([do_mcnemar(results, *pair) for pair in pairs]).set_index(['model1', 'model2'])
        save_pvalues(rr, run_folder, 'model_comp-model_accuracy_nemar')
        save_statistics(rr, run_folder, 'model_comp-model_accuracy_nemar')
    except:
        pass
    
    try:
        rr = pd.DataFrame([do_diebold_mariano(results, *pair) for pair in pairs]).set_index(['model1', 'model2'])
        save_pvalues(rr, run_folder, 'model_comp-diebold_mariano')
        save_statistics(rr, run_folder, 'model_comp-diebold_mariano')
    except:
        pass
    
    
    rr = pd.DataFrame([do_ttest(results, *pair, alpha = 1.0) for pair in pairs]).set_index(['model1', 'model2'])
    save_pvalues(rr, run_folder, 'model_comp-total_mean_ttest')
    save_statistics(rr, run_folder, 'model_comp-total_mean_ttest')
    
    rr = pd.DataFrame([do_ttest_paired(results, *pair, alpha = 1.0) for pair in pairs]).set_index(['model1', 'model2'])
    save_pvalues(rr, run_folder, 'model_comp-total_mean_ttest_paired')
    save_statistics(rr, run_folder, 'model_comp-total_mean_ttest_paired')
    
    rr = pd.DataFrame([do_wilcoxon_signed_rank(results, *pair) for pair in pairs]).set_index(['model1', 'model2'])
    save_pvalues(rr, run_folder, 'model_comp-total_median_wilcoxon')
    save_statistics(rr, run_folder, 'model_comp-total_median_wilcoxon')
    
    rr = pd.DataFrame([do_mannwhitneyu(results, *pair) for pair in pairs]).set_index(['model1', 'model2'])
    save_pvalues(rr, run_folder, 'model_comp-top-flop_median_mannwhitneyu')
    save_statistics(rr, run_folder, 'model_comp-top-flop_median_mannwhitneyu')
    
    rr = pd.DataFrame([do_welcht(results, *pair) for pair in pairs]).set_index(['model1', 'model2'])
    save_pvalues(rr, run_folder, 'model_comp-top-flop_mean_welcht')
    save_statistics(rr, run_folder, 'model_comp-top-flop_mean_welcht')
    
    rr = pd.DataFrame([do_ttest(results, *pair) for pair in pairs]).set_index(['model1', 'model2'])
    save_pvalues(rr, run_folder, 'model_comp-top-flop_mean_ttest')
    save_statistics(rr, run_folder, 'model_comp-top-flop_mean_ttest')
    
    
    
    feature_sets = list(results.loc[('8 RF-D20-E5000', slice(None), slice(None))].index.values)
    feature_sets = [('8 RF-D20-E5000', p, f) for p, f in feature_sets if f != '']
    feature_sets = list(filter(lambda x: x[2] in ['VR', 'FE', 'UIQ', 'POL+32', 'FE+POL+UIQ+VR+54'], 
                               feature_sets))
    pairs = list(itertools.product(feature_sets, feature_sets))
    pairs = [p for p in pairs if (p[0][1] == p[1][1])]
    
    
    try:
        rr = pd.DataFrame([do_mcnemar(results, *pair) for pair in pairs]).set_index(['model1', 'model2'])
        save_pvalues(rr, run_folder, 'feature_comp-model_accuracy_nemar')
        save_statistics(rr, run_folder, 'feature_comp-model_accuracy_nemar')
    except:
        pass
    
    try:
        rr = pd.DataFrame([do_diebold_mariano(results, *pair) for pair in pairs]).set_index(['model1', 'model2'])
        save_pvalues(rr, run_folder, 'feature_comp-diebold_mariano')
        save_statistics(rr, run_folder, 'feature_comp-diebold_mariano')
    except:
        pass
    
    rr = pd.DataFrame([do_ttest(results, *pair, alpha = 1.0) for pair in pairs]).set_index(['model1', 'model2'])
    save_pvalues(rr, run_folder, 'feature_comp-total_mean_ttest')
    save_statistics(rr, run_folder, 'feature_comp-total_mean_ttest')
    
    rr = pd.DataFrame([do_wilcoxon_signed_rank(results, *pair) for pair in pairs]).set_index(['model1', 'model2'])
    save_pvalues(rr, run_folder, 'feature_comp-total_median_wilcoxon')
    save_statistics(rr, run_folder, 'feature_comp-total_median_wilcoxon')
    
    rr = pd.DataFrame([do_mannwhitneyu(results, *pair) for pair in pairs]).set_index(['model1', 'model2'])
    save_pvalues(rr, run_folder, 'feature_comp-top-flop_median_mannwhitneyu')
    save_statistics(rr, run_folder, 'feature_comp-top-flop_median_mannwhitneyu')
    
    rr = pd.DataFrame([do_welcht(results, *pair) for pair in pairs]).set_index(['model1', 'model2'])
    save_pvalues(rr, run_folder, 'feature_comp-top-flop_mean_welcht')
    save_statistics(rr, run_folder, 'feature_comp-top-flop_mean_welcht')
    
    rr = pd.DataFrame([do_ttest(results, *pair) for pair in pairs]).set_index(['model1', 'model2'])
    save_pvalues(rr, run_folder, 'feature_comp-top-flop_mean_ttest')
    save_statistics(rr, run_folder, 'feature_comp-top-flop_mean_ttest')
    
    
#    ind_n = list(ex_results.index.names)
#    ind_n[-1] = 'kpi'
#    ex_results.index.names = ind_n
#    
#    uex_results = ex_results.unstack(level = [1, 0])
#    
#    SR = uex_results[[c for c in uex_results.columns.values if 'EP-ratio' not in c[1]]]
#    SR = SR.loc[(['FR', 'FR+DIS+FE', 'POL+FR+DIS+FE+50'], )]
#    SR.sort_index(axis = 1, level = 0, inplace = True)
#    
#    def fmter(x):
#        if pd.isna(x):
#            return '--'
#        else:
#            return '%.2f' % x if abs(x) < 1000 else '%d' % x
#    
#    L = SR.to_latex(float_format = fmter, multicolumn = True)
#    L = re.sub('[ ]+', ' ', L)
#    with open(run_folder + '/k.tex', "w") as f:
#        print(L, file = f)
    
    
    



if __name__ == '__main__':
    for run_folder in FINAL_RUN_FOLDERS:
        process_run(run_folder)



