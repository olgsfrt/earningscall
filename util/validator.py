#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#import pandas as pd

""""
class PhysicalTimeForwardValidation:
    
    def __init__(self, first_test_date, test_window_length, n_validation_samples, datetime_col):
        self.first_test_date = pd.Timestamp(first_test_date)
        self.test_window_length = pd.Timedelta(test_window_length)
        self.n_validation_samples = int(n_validation_samples)
        self.datetime_col = datetime_col
    
    def split(self, X):
        #assert type(X.index) is pd.Int64Index
        assert type(X.index) is pd.core.indexes.numeric.Int64Index

        
        X.sort_values(self.datetime_col, inplace = True)
        
        cur_test_start = self.first_test_date
        
        while X[self.datetime_col].max() > cur_test_start:
            i_train_end = X.index.get_loc(X.loc[X[self.datetime_col] 
                        < cur_test_start].iloc[-1].name) - self.n_validation_samples
            
            i_val_end = i_train_end + self.n_validation_samples
    
            i_test_end = X.index.get_loc(X.loc[X[self.datetime_col] 
                        < cur_test_start + self.test_window_length].iloc[-1].name)
            
            # print(i_train_end, i_val_end, i_test_end)
            
            train_index = X.index[0: i_train_end +1]
            val_index = X.index[i_train_end +1: i_val_end +1]
            test_index = X.index[i_val_end +1: i_test_end +1]
            
            cur_test_start = cur_test_start + self.test_window_length
            
            yield train_index, test_index, val_index
        
    def __str__(self):
        return "PhysicalTimeForwardValidation(%s, %s, %d, %s)" % (
                self.first_test_date, self.test_window_length,
                self.n_validation_samples, self.datetime_col)
        


if __name__ == "__main__":
    X = data
    
    val = PhysicalTimeForwardValidation('2012-01-01', pd.Timedelta(365, 'D'), 
                                        1000, 'final_datetime')
    
    for i, (train_index, test_index, val_index) in enumerate(val.split(X)):
        
        train_data = X.loc[train_index]
        test_data = X.loc[test_index]
        val_data = X.loc[val_index]
        
        print("Split #%d" % i)
        print("Train data of length %d from %s to %s" % (len(train_data), 
                            train_data.final_datetime.min(), train_data.final_datetime.max()))
        print("Validation data of length %d from %s to %s" % (len(val_data), 
                            val_data.final_datetime.min(), val_data.final_datetime.max()))
        print("Test data of length %d from %s to %s" % (len(test_data), 
                            test_data.final_datetime.min(), test_data.final_datetime.max()))
    


import pandas as pd
import numpy as np

class PhysicalTimeForwardValidation:
    
    def __init__(self, first_test_date, test_window_length, n_validation_samples, datetime_col):
        self.first_test_date = pd.Timestamp(first_test_date)
        self.test_window_length = pd.Timedelta(test_window_length)
        self.n_validation_samples = int(n_validation_samples)
        self.datetime_col = datetime_col
    
    def split(self, X):
        # check if the index is numeric (integer)
        assert np.issubdtype(X.index.dtype, np.integer)
        
        X.sort_values(self.datetime_col, inplace = True)
        
        cur_test_start = self.first_test_date

        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])

        
        while X[self.datetime_col].max() > cur_test_start:
            i_train_end = X.index.get_loc(X.loc[X[self.datetime_col] 
                        < cur_test_start].iloc[-1].name) - self.n_validation_samples
            
            i_val_end = i_train_end + self.n_validation_samples
    
            i_test_end = X.index.get_loc(X.loc[X[self.datetime_col] 
                        < cur_test_start + self.test_window_length].iloc[-1].name)
            
            train_index = X.index[0: i_train_end +1]
            val_index = X.index[i_train_end +1: i_val_end +1]
            test_index = X.index[i_val_end +1: i_test_end +1]
            
            cur_test_start = cur_test_start + self.test_window_length
            
            yield train_index, test_index, val_index
        
    def __str__(self):
        return "PhysicalTimeForwardValidation(%s, %s, %d, %s)" % (
                self.first_test_date, self.test_window_length,
                self.n_validation_samples, self.datetime_col)

if __name__ == "__main__":
    X = data  # make sure data is defined
    
    val = PhysicalTimeForwardValidation('2012-01-01', pd.Timedelta(365, 'D'), 
                                        1000, 'final_datetime')
    
    for i, (train_index, test_index, val_index) in enumerate(val.split(X)):
        
        train_data = X.loc[train_index]
        test_data = X.loc[test_index]
        val_data = X.loc[val_index]
        
        print("Split #%d" % i)
        print("Train data of length %d from %s to %s" % (len(train_data), 
                            train_data.final_datetime.min(), train_data.final_datetime.max()))
        print("Validation data of length %d from %s to %s" % (len(val_data), 
                            val_data.final_datetime.min(), val_data.final_datetime.max()))
        print("Test data of length %d from %s to %s" % (len(test_data), 
                            test_data.final_datetime.min(), test_data.final_datetime.max()))
"""

import pandas as pd
import numpy as np
import os

os.chdir('c:\\Users\\aq75iwit\\Anaconda3\\envs\\earnings_call_7\\EarningsCall')

class PhysicalTimeForwardValidation:
    
    def __init__(self, first_test_date, test_window_length, n_validation_samples, datetime_col):
        self.first_test_date = pd.Timestamp(first_test_date)
        self.test_window_length = pd.Timedelta(test_window_length)
        self.n_validation_samples = int(n_validation_samples)
        self.datetime_col = datetime_col
    
    def split(self, X):
        # check if the index is numeric (integer)
        assert np.issubdtype(X.index.dtype, np.integer), "Index is not numeric/integer"
        
        X.sort_values(self.datetime_col, inplace=True)
        
        cur_test_start = self.first_test_date
        
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        
        while X[self.datetime_col].max() > cur_test_start:
            i_train_end = X.index.get_loc(X.loc[X[self.datetime_col] < cur_test_start].iloc[-1].name) - self.n_validation_samples
            i_val_end = i_train_end + self.n_validation_samples

            i_test_end = X.index.get_loc(X.loc[X[self.datetime_col] < cur_test_start + self.test_window_length].iloc[-1].name)
            
            train_index = X.index[: i_train_end + 1]
            val_index = X.index[i_train_end + 1: i_val_end + 1]
            test_index = X.index[i_val_end + 1: i_test_end + 1]
            
            cur_test_start += self.test_window_length

            print('train_index:',train_index)
            
            yield train_index, test_index, val_index
        
    def __str__(self):
        return "PhysicalTimeForwardValidation(%s, %s, %d, %s)" % (
                self.first_test_date, self.test_window_length,
                self.n_validation_samples, self.datetime_col)


"""
if __name__ == "__main__":
    # Generate Sample Data for testing
    np.random.seed(42)
    n_samples = 5000
    date_rng = pd.date_range(start='2010-01-01', periods=n_samples, freq='D')
    X = pd.DataFrame(date_rng, columns=['final_datetime'])
    X['data'] = np.random.randint(0, 100, size=(n_samples))

    val = PhysicalTimeForwardValidation('2012-01-01', pd.Timedelta(365, 'D'), 
                                        1000, 'final_datetime')
    
    for i, (train_index, test_index, val_index) in enumerate(val.split(X)):
        
        train_data = X.loc[train_index]
        test_data = X.loc[test_index]
        val_data = X.loc[val_index]
        
        print("Split #%d" % i)
        print("Train data of length %d from %s to %s" % (len(train_data), 
                            train_data.final_datetime.min(), train_data.final_datetime.max()))
        print("Validation data of length %d from %s to %s" % (len(val_data), 
                            val_data.final_datetime.min(), val_data.final_datetime.max()))
        print("Test data of length %d from %s to %s" % (len(test_data), 
                            test_data.final_datetime.min(), test_data.final_datetime.max()))
        print("-----")
"""