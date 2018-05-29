#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:14:54 2018

@author: aditya
"""

import numpy as np
import pandas as pd
from scipy.stats import skewtest, boxcox

import os
import sys
root = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
## for use when in the interpreter
#root = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects')
sys.path.append(root)

from clean import mixedCategoricalClean, mixed2float

from sklearn.preprocessing import StandardScaler



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
    for i in range(m):
        current = data[i]
        date = '20' + current[5:7] + '-' + monthDict[current[2:5]] + '-' + current[:2]
        time = current[8:lastTimeIndex]
        newTime[i] = date + 'T' + time
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
        select[:,i] = arr == val
    select = np.any(select, axis = 1)
    
    if inverse:
        select = np.logical_and(~(arr == 'nan'), ~select)
    data[select] = newVal
    return data



def fixUrineVars(data):
    m = len(data)
    conversionDict = {'none': 'negative', 'few': 'trace', 'small': '1+', 'positive': '1+',
                  'moderate': '2+', 'large': '3+', '4+': '3+', 'many': '3+'}
    codings = ['negative', 'trace', '1+', '2+', '3+']
    basicConversion = lambda val: val if val in codings else conversionDict[val]
    

        
    # conversionUrineVars= ['uabili', 'uaprotein', 'uaglucose', 'uawbcs', 'uarbcs', 'uahycasts',
    #             'uanitrite', 'uaketones', 'ualeukest']
    # otherUrineVars = ['uaclarity', 'uacolor']
    
    # first do conversionUrineVars
    if 'uabili' in data.columns:
        col = data['uabili'].values.astype(np.str)
        select = ~(col == 'nan')
        new = np.empty(m, dtype = np.str)
        for i in range(m):
            if select[i]:
                try:
                    val = basicConversion(col[i])
                except KeyError:
                    try:
                        val = float(val)
                        if val == 0:
                            val = 'negative'
                        elif val <= 0.2:
                            val = 'trace'
                        elif val <= 1:
                            val = '1+'
                        elif val <= 2:
                            val = '2+'
                        else:
                            val = '3+'
                    except ValueError:
                        val = 'nan'
    
    
    # now do otherUrineVars    
    if 'uaclarity' in data.columns:
        data['uaclarity'] = mixedCategoricalClean(data['uaclarity'].values)
        data['uaclarity'] = stringCollapse(data['uaclarity'].values, ['clear'], 'cloudy', inverse = True)
    
    if 'uacolor' in data.columns:
        data['uacolor'] = mixedCategoricalClean(data['uacolor'].values)
        collapseList = ['yellow', 'colorless', 'pale yellow']
        data['uacolor'] = stringCollapse(data['uacolor'].values, collapseList, 'normal')
        data['uacolor'] = stringCollapse(data['uacolor'].values, ['normal'], 'abnormal', inverse = True)
    
    
    return data
    
    
    mixedCategorical(data[otherUrineVars[0]].values)

    
####################################################################################
# Code here needs revision and refactoring
# It was written a while ago
# Make the functions into classes in a scikit-learn style


def standardize(data, train_index, exclude = []):
    vars_to_standardize = list(set(data.columns) - set(data.columns[data.dtypes == np.bool]) - set(exclude))
    scaler = StandardScaler()
    data.loc[train_index, vars_to_standardize] = scaler.fit_transform(data.loc[train_index, vars_to_standardize])
    data.loc[~train_index, vars_to_standardize] = scaler.transform(data.loc[~train_index, vars_to_standardize])
    return data



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