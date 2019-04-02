#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:14:54 2018

@author: aditya
"""

import numpy as np
import pandas as pd
from scipy.stats import skewtest, boxcox, mode
from tqdm import tqdm

import os
import sys
root = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
## for use when in the interpreter
#root = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects')
sys.path.append(root)

from Helper.clean import mixedCategoricalClean, mixed2float

from sklearn.preprocessing import StandardScaler

class Splitter(object):
    """
    Splits data into training, val, test, whatever sets based on proportions arg
    proportion arg should sum to 1 and its length should correspond to how many sets you want
    random arg specifies if you want to initialize on a random split
    Holds splitting info in memory so sequential calls to split are aligned
    Can create new splits (partial refresh) by using shuffle method
    """
    def __init__(self, proportions, random = True):
        self.setProportions(proportions)
        self.shuffleBool = random
        self.n = None
        self.select = None
        
    def split(self, *data):
        
        # set-up for multiple or single inputs
        if len(data) > 1:
            singleInput = False
            n = len(data[0])
            # make sure all data provided is same length in first dim
            assert np.all(tuple(len(D) == n for D in data))
        else:
            singleInput = True
            data = data[0] # remove tuple wrapping
            n = len(data)
        
        # first-time init for self args n and select
        # else assert data is same length as previously established
        if self.needsInit:
            self.n = n
            self.select = self.generateSelector() # returns unshuffled
            if self.shuffleBool:
                self.shuffle()
            self.needsInit = False
        else:
            assert self.n == n
        
        # return split data for single or multiple inputs
        if singleInput:
            return tuple(data[self.select == i,...] for i in range(self.k))
        else:
            return tuple(tuple(D[self.select == i,...] for D in data) for i in range(self.k))
        
    def generateSelector(self):
        ## fill select with value i for ith partition
        ## note 0's are filled in already, so iterator starts at 1
        select = np.zeros(self.n, dtype = np.int8)
        for i in range(1, self.k):
            start = int(np.round(self.props[i-1]*self.n))
            stop = int(np.round(self.props[i]*self.n))
            select[start:stop] = i
        return select
    
    def shuffle(self):
        np.random.shuffle(self.select)
    
    def setProportions(self, proportions):
        TINY = 1e-5
        assert 1 - TINY <= sum(proportions) <= 1 + TINY
        self.props = np.cumsum(proportions)
        self.k = len(proportions)
        assert self.k >= 2
        self.needsInit = True
        

############ example usage
#X = data[features].values
#Y = data[outcome].values
#proportions = [0.7, 0.3]
#splitter = Splitter(proportions)
### use either
#(Xtrain, Ytrain), (Xtest, Ytest) = splitter.split(X, Y)
### or
#Xtrain, Xtest = splitter.split(X)
#Ytrain, Ytest = splitter.split(Y)


class Transformer(object):
    """
    Imputes using medians/modes (categoricals auto-inferred)
    Scales non-categoricals to similar ranges using regular standarization 
            (will upgrade to robust in future)
    """
    def fit(self, data):
        ## input data should be a numpy ndarray
        
        # recorded stats/information from each of k features
        self.k = np.shape(data)[1]
        self.hasMissing = np.zeros(self.k, dtype = np.bool)
        self.valueExists = np.zeros(self.k, dtype = np.bool)
        self.categorical = np.zeros(self.k, dtype = np.bool)
        self.values = np.zeros(self.k, dtype = np.float64)
        self.means = np.zeros(self.k, dtype = np.float64)
        self.stds = np.zeros(self.k, dtype = np.float64)
        
        for i in range(self.k):
            select = np.isfinite(data[:,i])
            dataslice = data[select,i]
            
            # find if we should impute at all
            # by examining if everything or nothing recorded
            self.hasMissing[i] = not np.all(select)
            self.valueExists[i] = not np.all(~select)
            # find out if categorical variable
            searchtill = np.minimum(len(dataslice), 20000)
            self.categorical[i] = len(np.unique(dataslice[:searchtill])) < 10
            
            # record mode for categoricals and medians for everything else
            if self.hasMissing[i] and self.valueExists[i]:
                if self.categorical[i]:
                    self.values[i] = mode(dataslice)
                else:
                    self.values[i] = np.median(dataslice)
            
            # record means and stds for non-categoricals
            if self.valueExists[i] and not self.categorical[i]:
                self.means[i] = np.mean(dataslice)
                std = np.std(dataslice)
                std = 1 if std == 0 else std
                self.stds[i] = std
    
    def transform(self, data):
        for i in range(self.k):
            if self.hasMissing[i] and self.valueExists[i]: # should impute?
                select = np.isfinite(data[:,i])
                data[~select,i] = self.values[i]
            if self.valueExists[i] and not self.categorical[i]:
                data[:,i] = (data[:,i]- self.means[i])/self.stds[i]
        return data
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
        
        


def oneHot(data, prefix = None):
    """
    input: (1-d) pandas series, prefix specifies custom prefix for column names
    output: (2-d) pandas dataframe (NaNs will be Falses across all columns)
    converts array into a matrix of binary markers for each unique value
    if only 2 unique values are found, returns single marker for first value (since second is redundant)
    """
    m = len(data)
    # discover markers
    limit = min(m, 100000)
    dataRestricted = data[:limit]
    names = list(np.unique(dataRestricted[np.isfinite(dataRestricted)]))
    
    # adjust for 1 col if data is truly binary
    if len(names) == 2:
        names = names[0]
    
    # actually do the one-hot encoding
    n = len(names)
    encoded = np.zeros((m,n), dtype = np.bool)
    for i in range(n):
        encoded[:,i] = data == names[i]
    
    # return with appropriate column names and index
    prefix = data.name + '_' if None else prefix
    return pd.DataFrame(encoded, index = data.index, columns = [prefix + name for name in names])



def convertISO8601(data, includeSec = False):
    """
    Input: 1-d numpy array filled with strings
    converts from: current format used in either EPIC or SAS [not sure] (29JUN18:21:59:55)
    convert to: ISO-8601, standardized datetime format that numpy accepts (2018-06-29T21:59:55)
    takes aroud 20sec for 10mil vals, so probably fine as current inefficient implementation
    """
    keys = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    values = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] 
    monthDict = dict(zip(keys,values))
    
    m = len(data)
    newTime = np.zeros(m, dtype = data.dtype)
    lastTimeIndex = 16 if includeSec else 13
    select = ~(data.astype(np.str) == 'nan')
    for i in range(m):
        if select[i]:
            current = data[i]
            try:
                date = '20' + current[5:7] + '-' + monthDict[current[2:5]] + '-' + current[:2]
                time = current[8:lastTimeIndex]
                newTime[i] = date + 'T' + time
            except (TypeError, KeyError):
                raise TypeError('Bad argument: ' + str(current)) 
        else:
            newTime[i] = ''
    return newTime.astype(np.datetime64)

def stringCollapse(data, collapseList, newVal, inverse = False):
    """
    Purpose: takes numpy array and finds all instances where the value is a member in collapseList
           if inverse is False: turns all those instances into newVal
           if inverse is True: turns all recorded non-instances into newVal
    Returns: modified array
    Note: assumes array is of (effectively) strings
    """
    select = np.zeros((len(data), len(collapseList)), dtype = np.bool)
    arr = data.astype(np.str)
    for i, val in enumerate(collapseList):
        select[:,i] = arr == str(val)
    select = np.any(select, axis = 1)
    
    if inverse:
        select = np.logical_and(~(arr == 'nan'), ~select)
    data[select] = newVal
    return data


    
####################################################################################
# Code here needs revision and refactoring
# It was written a while ago
# Make the functions into classes in a scikit-learn style

#
#def standardize(data, train_index, exclude = []):
#    vars_to_standardize = list(set(data.columns) - set(data.columns[data.dtypes == np.bool]) - set(exclude))
#    scaler = StandardScaler()
#    data.loc[train_index, vars_to_standardize] = scaler.fit_transform(data.loc[train_index, vars_to_standardize])
#    data.loc[~train_index, vars_to_standardize] = scaler.transform(data.loc[~train_index, vars_to_standardize])
#    return data



class SkewCorrection(object):
    def __init__(self, p = 0.05, copy = True):
        super().__init__()
        self.p = p
        self.copy = copy
        self.fitted = False
        
        
    def fit(self, data):
        data = data.copy() if self.copy else data
        n = np.shape(data)[1]
        
        skewed_bool = [skewtest(data[:,i])[1] < self.p for i in range(n)]
        self.indices = np.array(list(range(n)))[skewed_bool]
        self.indices = list(self.indices)
        g = len(self.indices)
        self.lambdas, self.minVals, self.nonNegative = [None]*g , [None]*g , [None]*g   

        for i, index in enumerate(self.indices):
            self.__single_boxcox_fit(data[:,index], i)
        self.fitted = True
        
    def transform(self, data):
        assert self.fitted
        data = data.copy() if self.copy else data
        for i, index in enumerate(self.indices):
            data[:,index] = self.__single_boxcox_transform(data[:, index], i)
        return data
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
        
    def __single_boxcox_fit(self, col, index):
        
        self.minVals[index] = np.min(col)
        self.nonNegative[index] = True if self.minVals[index] >= 0 else False
        col = self.__minTrans(self.nonNegative[index], col, self.minVals[index])
        
        _, lmbda = boxcox(col, lmbda = None)
        self.lambdas[index] = lmbda
        
    def __single_boxcox_transform(self, col, index):
        col = self.__minTrans(self.nonNegative[index], col, self.minVals[index])
        result = boxcox(col, lmbda = self.lambdas[index])
        return result
        
    def __minTrans(self, nonNegative, col, minVal):
        if nonNegative == True:
            col = col - minVal + 20
        else:
            col = col - 4*minVal + 20
        return col