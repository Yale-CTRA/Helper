#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:12:08 2018

@author: aditya
"""

import numpy as np
import pandas as pd
from copy import deepcopy, copy
import warnings
import os
import sys
from tqdm import tqdm
root = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
sys.path.append(root)

from Helper.preprocessing import SkewCorrection
from sklearn.preprocessing import Imputer, StandardScaler



class PanelIndexer(object):
    def __init__(self, data, idname, timename, verbose = True):
        self.data = data.sort_values([idname, timename])
        self.data.set_index(idname, inplace = True, drop = False)
        self.idname = idname
        self.timename = timename
        self.verbose = verbose
        self.pIndex = self.getpIndex(self.data[self.idname].values)
    
    def getpIndex(self, data):
        m = len(data)
        X = np.zeros((len(np.unique(data)), 3), dtype = np.int64)
        #fix boundaries
        X[0,0] = 0
        X[-1,1] = m
        # do loop to record indices
        counter = 0
        current = data[0]
        for i in range(1,m):
            if data[i] != current:
                current = data[i]
                X[counter,1] = i
                X[counter+1,0] = i
                counter += 1
        X[:,2] = X[:,1] - X[:,0]
        return X
    
    def subset(self, select):
        self.data = self.data.loc[select,:]
        self.pIndex = self.getpIndex(self.data[self.idname].values) 
    
    
    def create(self, func, ncols = 1, dtype = np.float64, by = 'row'):
        assert by in ('encounter', 'row', 'panel')
        shape = len(self.pIndex) if by == 'panel' else len(self.data)
        shape = (shape, ncols) if ncols > 1 else shape
        new = np.empty(shape, dtype = dtype)
        if by == 'encounter':
            for i in tqdm(range(len(self.pIndex))):
                start, stop, length = self.pIndex[i,:]
                new[start:stop] = func(self.data.iloc[start:stop,:])
        elif by == 'row':
            new[:] = func(self.data)
        elif by == 'panel':
            for i in tqdm(range(len(self.pIndex))):
                start, stop, length = self.pIndex[i,:]
                new[i] = func(self.data.iloc[start:stop,:])
        return new
    
    def filter(self, keepFunc, by = 'row', exclude = False):
        assert by in ('panel', 'row', 'encounter')
        m0 = len(self.data) if by == 'row' else len(self.pIndex)
        if by == 'panel':
            func = lambda x: np.repeat(keepFunc(x), len(x))
            select = self.create(func, dtype = np.bool, by = 'encounter')
        elif by == 'encounter':
            select = self.create(keepFunc, dtype = np.bool, by = 'encounter')
        elif by == 'row':
            select = keepFunc(self.data)
            
        select = np.logical_not(select) if exclude else select
        self.subset(select)
        
        if self.verbose:
            # print what occurred
            if by == 'encounter':
                m1 = self.countAllTrue(select)
            elif by == 'panel':
                m1 = len(self.pIndex)
            else:
                m1 = len(self.data)
            nounDict = {'row': 'row', 'encounter': 'encounter', 'panel': 'encounter'}
            verbDict = {'row': 'removed', 'encounter': 'shortened', 'panel': 'removed'}
            print(m0-m1, nounDict[by] + 's', verbDict[by])
            if by == 'encounter':
                print(np.sum(~select), 'rows removed')
    
    def countAllTrue(self, select):
        count = 0
        for i in range(len(self.pIndex)):
            start, stop, _ = self.pIndex[i,:]
            count += np.all(select[start:stop])
        return count
        
    def keepfirstN(self, N):
        # returns first n encounters
        return PanelIndexer(self.data.iloc[:self.pIndex[N-1,1],:], self.idname, self.timename)
    
    def split(self, percentage):
        cutoff = self.pIndex[int(np.round(len(self.pIndex)*percentage)),0]
        return (PanelIndexer(self.data.iloc[:cutoff,:], self.idname, self.timename),
                PanelIndexer(self.data.iloc[cutoff:,:], self.idname, self.timename))

    
    def drop(self, labels, axis = 1):
        self.data.drop(labels = labels, axis = axis, inplace = True)
        
    def impute(self, label):
        avgSigma = np.nanmean(self.create(lambda x: np.nanmean(x[label].values),
                                   dtype = np.float32, by = 'panel'))
        select = np.isnan(self.data[label].values)
        self.data.loc[select,label] = avgSigma
        
        if self.verbose:
            print(len(self.pIndex) - self.countAllTrue(~select), 'encounters needed imputation')
            print(np.sum(select), 'values imputed')
    
        
    def __len__(self):
        return len(self.pIndex)
    
    def __getitem__(self, key):
        if type(key) is int:
            start, stop, _ = self.pIndex[key,:]
            return self.data.iloc[start:stop,:]
        elif type(key) is slice:
            pIndexSub = self.pIndex[key,:]
            start, stop = pIndexSub[0,0], pIndexSub[-1,1]
            return self.data.iloc[start:stop,:]
        elif type(key) is str or list: 
            return self.data[key]
        else:
            raise KeyError("Use a slice, int, or str as the key")
    
    def __setitem__(self, keys, values):
        if type(keys) is list:
            for i, name in enumerate(keys):
                self.data[name] = values[:,i]
        elif type(keys) is str:
            self.data[keys] = values
        else:
            raise KeyError("Use a str or list of strings as the key")
    




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
        self.df = df.astype(np.float32)
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
        self.__trainIndex = self.buildIndexer(start = 0, stop = split[0])
        self.__valIndex = self.buildIndexer(start = split[0], stop = sum(split[:2]))
        self.__testIndex = self.buildIndexer(start = sum(split[:2]), stop = 1)
        self.refresh()
        
        
    def refresh(self, seed = None, randomize = True):
        np.random.seed(seed=seed)
        df = self.df.sample(frac = 1.) if randomize else self.df
        self.train, self.val, self.test = {}, {}, {}
        for key in self.info:
            self.train[key] = df.loc[self.__trainIndex, self.info[key]].values
            self.val[key] = df.loc[self.__valIndex, self.info[key]].values
            self.test[key] = df.loc[self.__testIndex, self.info[key]].values
            
            


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
