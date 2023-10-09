# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 17:40:21 2023

@author: aq75iwit
"""
"""
import pickle
import os
import logging
import datetime
import pandas as pd
#import numpy as np
#import importlib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import keras

from util.validator import PhysicalTimeForwardValidation
#from util.prepare_data import prepare_data, clean_data

from learning_model import run_model, summarize_model_results

from config import RUNS_FOLDER

import json



print(os.getcwd())


import pickle
import os
import logging
import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras  # Changed this line
from util.validator import PhysicalTimeForwardValidation
from learning_model import run_model, summarize_model_results
from config import RUNS_FOLDER
import json
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScale

import tensorflow as tf
print(tf.__version__)
"""

import pickle
import os
import logging
import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from util.validator import PhysicalTimeForwardValidation
from learning_model import run_model, summarize_model_results
from config import RUNS_FOLDER
import json
from scikeras.wrappers import KerasClassifier # Changed this line

#with open('E:/data/Regressionen/all/data.json', encoding='utf-8') as data_file:
    #dict_list = json.loads(data_file.read())
    

#data=pd.DataFrame(dict_list)

data=pd.read_pickle('D:/01_Diss_Data/00_Data_Final/merged_data_17_05_2023.pickle')

JOB_CONFIG_FILE = 'final'
data = data.sort_values('final_datetime')
data.reset_index(drop=True, inplace=True)



def create_model():
    from tensorflow.keras.regularizers import l2
    
    classifier = keras.Sequential()
    
    classifier.add(keras.layers.Dense(15, input_dim = 15, activation = 'relu'))
    classifier.add(keras.layers.Dense(4, activation = 'relu', activity_regularizer=l2(1E-3)))
    classifier.add(keras.layers.Dense(2, activation = 'softmax'))
    
    opt = keras.optimizers.Nadam()
    
    classifier.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    
    return classifier

earl = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, min_delta = 0)

logdir = RUNS_FOLDER + "/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model = KerasClassifier(model=create_model, batch_size = 128, epochs = 200, verbose = 1) # Changed this line

model = Pipeline([('scaler', StandardScaler()), ('ANN', model)])


"""
job = {
       'train_subset': 'SP1500',
       'model': model,
       'train_target': 'abnormal_5d_drift',
       'return_target': 'abnormal_5d_drift',
       'features': ['earnings_surprise', 
                    'earnings_surprise_mean_std', 
                    'earnings_surprise_std', 
                    'earnings_surprise_revisions', 
                    'earnings_surprise_estimates', 
                    'earnings_ratio', 
                    'pays_dividend', 
                    'revenue_surprise', 
                    #'revenue_surprise_std', 
                    'revenue_surprise_estimates', 
                    #'same_day_call_count', 
                    #'hour_of_day_half', 
                    'log_length', 
                    'nr_analysts', #'nr_executives',
                    'general_PositivityLM', 'general_NegativityLM', 
                    'qanda_PositivityLM', 'qanda_NegativityLM', 
                    
                    #'general_SentimentLM', 'qanda_SentimentLM',
                    #'general_PositivityHE', 'general_NegativityHE', 
                    #'qanda_PositivityHE', 'qanda_NegativityHE', 
                    #'call_return',
                    #'-1d_pre-drift', '-2d_pre-drift', 
                    #'-4d_pre-drift',
                    #'-9d_pre-drift', '-20d_pre-drift',
                    #'es_drift_9', 'es_drift_5'
                    #'healthcare', 'industrial-goods', 'technology', 'financial', 'utilities', 
                    #'consumer-goods', 'services', 'basic-materials','conglomerates'
                    ], #+ ['%d_b'%c for c in [10, 25, 29]],
        'top_flop_cutoff': 0.1,
        'validator': PhysicalTimeForwardValidation('2013-01-01', pd.Timedelta(365, 'D'), 1500, 'final_datetime'),
        'rolling_window_size': 1500,
        'calculate_permutation_feature_importances': False
       }
"""
job = {
       'train_subset': 'SP1500',
       'model': model,
       'train_target': 'abnormal_5d_drift',
       'return_target': 'abnormal_5d_drift',
       'features': ['BM_ratio', 'EP_ratio', 'SP_ratio', 'CP_ratio', 
        'DY_ratio', 'dividend_payout_ratio','MV_log','BM_surprise', 'EP_surprise', 'SP_surprise', 
        'DY_surprise', 'CP_surprise','BM_surprise', 'EP_surprise', 'SP_surprise', 'DY_surprise', 'CP_surprise'], #+ ['%d_b'%c for c in [10, 25, 29]],
        'top_flop_cutoff': 0.1,
        'validator': PhysicalTimeForwardValidation('2013-01-01', pd.Timedelta(365, 'D'), 1500, 'final_datetime'),
        'rolling_window_size': 1500,
        'calculate_permutation_feature_importances': False
       }



def run_job(job_id, job, run_folder):
    model_summaries = []
    
    job_folder = run_folder + '/job-%d/' % job_id
    
    if not os.path.exists(job_folder):
        os.makedirs(job_folder)
    
    predictions, model_results = run_model(data, job)
    
    predictions.to_hdf(job_folder + 'predictions.hdf', 'predictions')
    pd.Series(model_results).to_hdf(job_folder + 'model_results.hdf', 'results')
    
    summary = summarize_model_results(job, model_results)
    model_summaries.append(pd.concat([pd.Series(job_id, name = 'job_id'), summary], 
                           axis = 0))
    
    return model_summaries

def save_summaries(model_summaries, folder):
    model_summaries = pd.concat(model_summaries, axis = 1).transpose()
    
    writer = pd.ExcelWriter(folder + 'summary.xlsx', engine = 'xlsxwriter')
    
    model_summaries.to_excel(writer, sheet_name = 'model summary')
    
    try:
        writer.save()
    except Exception as e:
        print('Error occurred during file writing:', e)

model_jobs = [job]

print(model_jobs)

runn = 'ann-run'
ts = datetime.datetime.now().replace(microsecond = 0).isoformat().replace(':', '_')
run_folder = RUNS_FOLDER + '/%s-%s/' % (runn if runn is not None else 'run', ts)

logging.info("Writing results to run folder %s", run_folder)

model_summaries = []

for job_id, job in enumerate(model_jobs):
    logging.info("Running job %d of %d", job_id, len(model_jobs))
    
    model_summary = run_job(job_id, job, run_folder)
    
    model_summaries += model_summary

save_summaries(model_summaries, run_folder)


for i in data.columns:
    if 'ff-dec' in i:
        print(i)
