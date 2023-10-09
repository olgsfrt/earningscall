#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:44:15 2019

@author: mschnaubelt
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('c:\\Users\\aq75iwit\\Anaconda3\\envs\\earnings_call_7\\EarningsCall')



def analyze_decile_return(result, output_basefilename = None):
    dr = result['deciles_returns']
    
    T = dr[['mean', 't']].apply(lambda r: '%.2f (%.2f)' % (r['mean']*1E4, r['t']), 
          axis = 1)

    #T = T.to_frame().append(pd.Series({'Top-flop abnormal return': '%.2f bp (%.2f)' % (result['top_flop_mean']*1E4, result['top_flop_t'])}).to_frame()).squeeze()
    #T = T.to_frame().T.append(pd.Series({'Top-flop abnormal return': '%.2f bp (%.2f)' % (result['top_flop_mean']*1E4, result['top_flop_t'])}).to_frame().T).squeeze()
    # Convert the Series T to a DataFrame
    #T_df = T.to_frame()

    # Create a new DataFrame with the data you want to append
    #new_data = {'Top-flop abnormal return': ['%.2f bp (%.2f)' % (result['top_flop_mean']*1E4, result['top_flop_t'])]}
    #new_data_df = pd.DataFrame(new_data, columns=T_df.columns)  # Ensure new DataFrame has same columns as T_df

    # Append the new data to T_df
    #T_df = T_df.append(new_data_df)

    # Convert the DataFrame back to a Series
    #T = T_df.squeeze()

    new_series = pd.Series({'Top-flop abnormal return': '%.2f bp (%.2f)' % (result['top_flop_mean']*1E4, result['top_flop_t'])})
    T = pd.concat([T, new_series])


    #T = T.append(pd.Series({'Top-flop abnormal return': '%.2f bp (%.2f)' % (result['top_flop_mean']*1E4, result['top_flop_t'])}))

    #T = T.append(pd.Series({'Top-flop abnormal return': '%.2f bp (%.2f)' % (result['top_flop_mean']*1E4, result['top_flop_t'])}))
    
    if 'test_bacc' in result:
        #T = T.append(pd.Series({'Top-flop BACC': '%.2f%%' % (result['top_flop_bacc']*100, )}))
        #T = T.append(pd.Series({'Overall BACC': '%.2f%%' % (result['test_bacc']*100, )}))
        top_flop_series = pd.Series({'Top-flop BACC': '%.2f%%' % (result['top_flop_bacc']*100, )})
        overall_bacc_series = pd.Series({'Overall BACC': '%.2f%%' % (result['test_bacc']*100, )})
        T = pd.concat([T, top_flop_series, overall_bacc_series])

    
    if output_basefilename:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.subplots(1, 1)
        
        ax.bar([str(i) for i in dr.index.values], dr['mean']*10000, 
               color = 'lightgray')
        
        for i, p in enumerate(ax.patches):
            ax.annotate("%.2f\n(%.2f)" % (dr.iloc[i]['mean']*10000, dr.iloc[i].t), 
                        (p.get_x() + p.get_width()/2, 40), fontsize = 10, ha = "center")
        
        ax.set_ylim(dr['mean'].min()*10000 - 10, dr['mean'].max()*10000 + 10)
        
        ax.set_ylabel('mean abnormal return [bp]', fontsize = 14)
        ax.set_xlabel('decile', fontsize = 14)
        
        ax.tick_params(axis = 'both', which = 'major', labelsize = 13)
        plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")
        
        fig.tight_layout()
        fig.savefig(output_basefilename + '.pdf')
        
        plt.close()
        
        T.to_csv(output_basefilename + '.csv')
    
    return T



if __name__ == '__main__':
    RUN_FOLDER = '/mnt/data/earnings_calls/runs/run-2019-11-12T15_15_13/'
    
    jobs = [f for f in os.listdir(RUN_FOLDER) if f.startswith('job')]
    jobs.sort(key = lambda s: int(s.replace('job-', '')))
    results = [pd.read_hdf(RUN_FOLDER + '%s/model_results.hdf' % f) for f in jobs]
    
    results = pd.concat(results, axis = 1)
    
    Ts = []
    
    for job_id, result in results.iteritems():
        print(job_id)
        T = analyze_decile_return(result, RUN_FOLDER + 'mean_return_job-%d' % job_id)
        Ts.append(T)
    
    Ts = pd.concat(Ts, axis = 1)
    
    Ts.to_csv(RUN_FOLDER + 'decile_returns.csv')


