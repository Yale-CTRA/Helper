#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:56:39 2018

@author: aditya
"""

import numpy as np



"""
functions to remove unwanted columns
"""
## returns dataframe with only numerical (bool, int, float) columns kept
def keepNumerical(data):
    dtype_set = {np.int8, np.int16, np.int32, np.float32, np.float64, np.bool}
    return data.loc[:,data.dtypes.apply(lambda x: any([issubclass(x.type, dtype) for dtype in dtype_set]))]

## returns dataframe with columns that are all NaNs removed
def removeNaNColumns(data):
    keep = [np.all(np.isnan(data.iloc[:,i])) for i in range(len(data.columns))]
    return data.loc[:,keep]


"""
functions to discover if conversions of type will preserve informaion
"""
## returns boolean array describing which variables are boolean
def isBool(df, varList = None):
    varList = list(df.columns) if varList is None else varList
    k = np.repeat(False, len(df)) # search restriction for large datasets to be faster
    k[:min(len(df), 10000)] = True
    return np.array([len(np.unique(df.loc[np.logical_and(np.isfinite(df[var].values),k),var])) <= 2 
                         for var in varList], dtype = np.bool)

## returns boolean array describing which variables are integers
def isInt(df, varList = None):
    df.fillna(0, inplace = True) # prevents error during comparison
    varList = varList if not None else list(df.columns)
    k = min(len(df), 10000) # search restriction for large datasets to be faster
    return np.array([np.all(df[var].iloc[:k].values == df[var].iloc[:k].values.astype(np.int)) for var in varList], dtype = np.bool)


"""
functions to compress numerical data types to smallest necessary
"""

## finds the smallest integer-type size for this array of ints
def optimalIntType(col):
    m, M = np.min(col), np.max(col)
    bound = max(abs(m), abs(M))
    signed = False if m >= 0 else True
    
    # cutoffs found online
    if signed:
        if bound <= 128:
            dtype = np.dtype('int8')
        elif bound <= 32768:
            dtype = np.dtype('int16')
        elif bound <= 2147483648:
            dtype = np.dtype('int32')
        else:
            dtype = np.dtype('int64')
    else:
        if bound <= 255:
            dtype = np.dtype('uint8')
        elif bound <= 65535:
            dtype = np.dtype('uint16')
        elif bound <= 4294967295:
            dtype = np.dtype('uint32')
        else:
            dtype = np.dtype('uint64')
    return dtype



## returns data with all columns compressed to as small of a datatype as possible
## assumes only numerical rows
def compress(data):
    new_dtypes = data.dtypes.to_dict()
    names = data.columns.values
    
    ## compress bool types
    boolIndices = isBool(data)
    boolNames = names[boolIndices]
    for name in list(boolNames):
        new_dtypes[name] = np.dtype('bool')
    remainingNames = names[np.logical_not(boolIndices)]
    
    ## compress int types
    intIndices = isInt(data, remainingNames)
    intNames = remainingNames[intIndices]
    for name in list(intNames):
        new_dtypes[name] = optimalIntType(data.loc[:,name].values)
    remainingNames = remainingNames[np.logical_not(intIndices)]

    ## compress ALL float types to float16: assumes all values are 
    ## much, much less than 65,536 (precision is very bad >32,768)
    for name in list(remainingNames):
        new_dtypes[name] = np.dtype('float16')
        
    return data.astype(new_dtypes)




