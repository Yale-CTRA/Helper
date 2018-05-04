#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:12:08 2018

@author: aditya
"""

import numpy as np
import pandas as pd
from copy import copy
import warnings
import sys
path = '/home/aditya/Projects/'
sys.path.append(path)
from Helper.preprocessing import SkewCorrection
from sklearn.preprocessing import Imputer, StandardScaler


class Data(object):
    """
    Inputs:
            1. Pandas Dataset
            2. Dictionary that maps keywords to variable sets (predictors, targets, etc...) 
            3. List with fractions of data that fall into training, validation, and test sets
    Notable Features:
            1. Refresh method automatically randomly splits dataset into train, val, and test sets and stores internally
            2. Less verbose access of variable groups (data.train['key'] vs data.train.loc[:,data.info['key']].values)
    Caveats: If imputation or scaling is necessary, it must be done beforehand on whole dataset (biased)
    """
    def __init__(self, df, info, split):
        self.df = df.astype(np.float16)
        self.m = len(df)
        self.info = info
        self.setSplit(split) # also refreshes
        
    def buildIndexer(self, start, stop): # start/stop refer to percentiles
        start = int(round(self.m*start))
        stop = int(round(self.m*stop))
        indexer = np.repeat(False, self.m)
        indexer[start:stop] = True
        return indexer
    
    def setSplit(self, split):
        assert sum(split) == 1
        self.trainIndex = self.buildIndexer(start = 0, stop = split[0])
        self.valIndex = self.buildIndexer(start = split[0], stop = sum(split[:2]))
        self.testIndex = self.buildIndexer(start = sum(split[:2]), stop = 1)
        self.refresh()
        
        
    def refresh(self, randomize = True):
        df = copy(self.df.sample(frac = 1.) if randomize else self.df)
        self.train, self.val, self.test = {}, {}, {}
        for key in self.info:
            self.train[key] = df.loc[self.trainIndex, self.info[key]].values
            self.val[key] = df.loc[self.valIndex, self.info[key]].values
            self.test[key] = df.loc[self.testIndex, self.info[key]].values
            
            


### this is a very old class i wrote for normal classification/regression outcomes
### does all imputation, skew-correction, and scaling in an unbiased way
### but this code probably needs a lot of revision and refactoring
    
class UpliftDataContainer(object):
    '''
    data:           pandas.DataFrame
    info:           dict = {'ID': 'id_string', 'predictors': ['predictor1', ...], 
                            'targets': ['target0', ...], 'treatment' = 'treat_string'}
    train_frac:     float (percentage of dataset you want to be training set)
    batch_size:     int (number of instances fed during iterator calls)
    '''
    def __init__(self, data, info, train_frac = 0.7, batch_size = 32):
        super().__init__()
        assert type(data) == pd.core.frame.DataFrame and type(info) == dict
        assert type(train_frac) == float and type(batch_size) == int
        self.train, self.val, self.test = None, None, None
        self.id, self.predictors, self.targets, self.weights = info['ID'], info['predictors'], info['targets'], info['weights']
        self.treatment = info['treatment']
        self.train_frac = train_frac
        self.train_index, self.val_index = self.__build_indexer(len(data), self.train_frac)
        self.batch_size = batch_size
        self.replacement = False

        self.data_original = self.__clean(data.set_index(self.id, drop = True, inplace = False))
        #if len(self.targets) == 1:
        #self.data_original['goodalert'] = classVariableTransform(
        #            self.data_original[self.targets].values, self.data_original[self.treatment].values)
        self.columns = self.data_original.columns.values
        self.vars_bool = self.__find_binary(self.data_original)
        self.vars_float = list(set(self.columns) - set(self.vars_bool))
        self.__set_predictor_groups()
        self.predictors_float = np.concatenate((self.predictors_float, np.array(['sbp_pca', 'dbp_pca'])))
        
        #self.refresh()
        self.train_mode()
            
    def __len__(self):
        return len(self.current)
        
    def __iter__(self):
        self.index = 0
        self.stop = False
        return self
    
    def __next__(self):
        end = self.index + self.batch_size
        if self.stop:
            raise StopIteration
        elif end >= len(self):
            self.stop = True
            batch = self.current.iloc[self.index:,:]
#            batch_X_bool = batch[self.predictors_bool].values.astype('float32')
#            batch_X_float = batch[self.predictors_float].values.astype('float32')
#            batch_Y = batch[self.targets].values.astype('float32')
#            batch_T = batch[self.treatment].values.astype('float32')
            return batch
        else:
            batch = self.current.iloc[self.index:end,:]
#            batch_X_bool = batch[self.predictors_bool].values.astype('float32')
#            batch_X_float = batch[self.predictors_float].values.astype('float32')
#            batch_Y = batch[self.targets].values.astype('float32')
#            batch_T = batch[self.treatment].values.astype('float32')
            self.index = end
            return batch
        
        
        
    def __impute(self, mode = None):
        assert (mode == 'most_frequent' or mode == 'mean')
        vars_now = self.predictors_bool if mode == 'most_frequent' else self.predictors_float
        imputer = Imputer(strategy = mode, copy = True)
        self.train.ix[:,vars_now] = imputer.fit_transform(self.train[vars_now].values)       
        self.test.ix[:,vars_now] = imputer.transform(self.test[vars_now].values)

    
    def __skew_correct(self, p = 0.05):
        skewCorrecter = SkewCorrection(p = p, copy = True)
        self.train.ix[:,self.predictors_float] = skewCorrecter.fit_transform(self.train[self.predictors_float].values)
        self.test.ix[:,self.predictors_float] = skewCorrecter.transform(self.test[self.predictors_float].values)
    
    def __scale(self):
        scaler = StandardScaler(copy = True)
        self.train.ix[:,self.predictors_float] = scaler.fit_transform(self.train[self.predictors_float].values)
        self.test.ix[:,self.predictors_float] = scaler.transform(self.test[self.predictors_float].values)

        
    def __build_indexer(self, m, train_per):
        train_index = np.repeat(False, m)
        val_index = np.repeat(False, m)
        cutoff = int(round(m*train_per))
        train_index[:cutoff] = True
        cutoff_val = cutoff + int((m - cutoff)/2)
        val_index[:cutoff_val] = True
        return train_index, val_index
    

    def __clean(self, data):
        # removes pure nan columns and strings
        dtype_set = {np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, np.bool}
        return data.loc[:,data.dtypes.apply(lambda x: any([issubclass(x.type, dtype) for dtype in dtype_set]))]

    def __find_binary(self, data):
        data = data.copy(deep = True).fillna(value = 0)
        binary_bool = [len(np.unique(data[col].values)) <= 10 for col in data.columns.values]
        return data.columns.values[binary_bool]
    
    def __set_predictor_groups(self):
        self.predictors_bool = np.array(self.vars_bool)[[var in self.predictors for var in self.vars_bool]]
        self.predictors_float = np.array(self.vars_float)[[var in self.predictors for var in self.vars_float]]
        
    
    def __pca(self):
        pca = PCA()
        temp1 = pca.fit_transform(self.train[['sbp', 'dbp']].values)
        self.train['sbp_pca'] = temp1[:,0]
        self.train['dbp_pca'] = temp1[:,1]
        
        temp2 = pca.transform(self.test[['sbp', 'dbp']].values)
        self.test['sbp_pca'] = temp2[:,0]
        self.test['dbp_pca'] = temp2[:,1]
        
#    def __upsample(self, train, num_times):
#        train_y = train.ix[train[self.targets].values[:,0] == 1, :]
#        train = pd.concat([train] + [train_y]*num_times, axis = 0)
#        return train
        
    def shuffle(self, data = None):
        if data is None:
            self.current = self.current.sample(frac = 1., replace = self.replacement, axis = 0)
        else:
            return data.sample(frac = 1., replace = self.replacement, axis = 0)
    
    def refresh(self, shuffle_bool = True, verbose = True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if verbose:
                print('Refreshing...')
            if shuffle_bool:
                data = self.shuffle(self.data_original.copy(deep = True))
                if verbose:
                    print('Data Shuffled.')
            else:
                data = self.data_original.copy(deep = True)
                
            self.train, self.test = data.iloc[self.train_index,:], data.iloc[~self.val_index,:]
            self.__pca()
            
            self.__impute(mode = 'most_frequent')
            if verbose:
                print('Discrete Variables Imputed.')
            self.__impute(mode = 'mean')
            if verbose:
                print('Continuous Variables Imputed.')
                
            
            self.__skew_correct(p = 0.05)
            if verbose:
                print('Data Skew Corrected.')
            self.__scale()
            if verbose:
                print('Data Scaled.')
            #train = self.__upsample(train, 6)
            
            #self.test = self.test.ix[self.test['t_primary'].values > 0, 't_primary']
            self.val = pd.concat([self.train, self.test], axis = 0).iloc[self.val_index,:]
            self.current = self.train
    
    
    def train_mode(self, verbose = False):
        self.current = self.train
        self.mode = 'train'
        if verbose:
            print('Entering training mode.')
    
    def val_mode(self, verbose = False):
        self.current = self.val
        self.mode = 'val'
        if verbose:
            print('Entering validation mode.')
    
    def test_mode(self, verbose = False):
        self.current = self.test
        self.mode = 'test'
        if verbose:
            print('Entering testing mode.')
        
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    def set_train_frac(self, train_frac):
        self.train_frac = train_frac
    def set_predictors(self, predictors):
        self.predictors = predictors
        self.__set_predictor_groups()
    def set_targets(self, targets):
        self.targets = targets
