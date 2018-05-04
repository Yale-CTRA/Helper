#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:14:54 2018

@author: aditya
"""

import numpy as np
import pandas as pd
from scipy.stats import skewtest, boxcox

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

"""
All code here needs revision and refactoring
It was written a while ago
Make the functions into classes in a scikit-learn style
"""


    
def standardize(data, train_index, exclude = []):
    vars_to_standardize = list(set(data.columns) - set(data.columns[data.dtypes == np.bool]) - set(exclude))
    scaler = StandardScaler()
    data.loc[train_index, vars_to_standardize] = scaler.fit_transform(data.loc[train_index, vars_to_standardize])
    data.loc[~train_index, vars_to_standardize] = scaler.transform(data.loc[~train_index, vars_to_standardize])
    return data



def one_hot(df, var_name):
    col = df[var_name].values.reshape(-1,1)
    all_values = np.unique(col)
    for i in range(len(all_values)):
        col[col == all_values[i]] = i
    oneHot = OneHotEncoder(dtype = 'int16')
    col_names = ['hispanic', 'black', 'white', 'other']
    col_one_hot = pd.DataFrame(oneHot.fit_transform(col).toarray(), index = df.index,
                           columns = col_names)
    df = pd.concat([df, col_one_hot], axis = 1)
    return df



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