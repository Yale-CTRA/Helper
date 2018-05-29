#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import re

#####################################################################################################
########## functions to remove unwanted columns

## returns dataframe with only numerical (bool, int, float) columns kept
def keepNumerical(data):
    dtype_set = {np.int8, np.int16, np.int32, np.float32, np.float64, np.bool}
    return data.loc[:,data.dtypes.apply(lambda x: any([issubclass(x.type, dtype) for dtype in dtype_set]))]

## returns dataframe with columns that are all NaNs removed
def removeNaNColumns(data):
    keep = [np.all(np.isnan(data.iloc[:,i])) for i in range(len(data.columns))]
    return data.loc[:,keep]


#####################################################################################################
######### functions to discover if conversions of type will preserve informaion

## returns boolean array describing which variables are boolean
def isBool(df, varList = None):
    varList = list(df.columns) if varList is None else varList
    k = np.repeat(False, len(df)) # search restriction for large datasets to be faster
    k[:min(len(df), 100000)] = True
    return np.array([len(np.unique(df.loc[np.logical_and(np.isfinite(df[var].values),k),var])) <= 2 
                         for var in varList], dtype = np.bool)

## returns boolean array describing which variables are integers
def isInt(df, varList = None):
    df.fillna(0, inplace = True) # prevents error during comparison
    varList = varList if not None else list(df.columns)
    k = min(len(df), 100000) # search restriction for large datasets to be faster
    return np.array([np.all(df[var].iloc[:k].values == df[var].iloc[:k].values.astype(np.int)) for var in varList], dtype = np.bool)


#####################################################################################################
############## functions to compress numerical data types to smallest necessary

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
    ## much, much less than 65,536 (precision is terrible ~32,768 or more, but very good close to 0)
    for name in list(remainingNames):
        new_dtypes[name] = np.dtype(np.float16)
        
    return data.astype(new_dtypes)


#####################################################################################################
##############  Functions for fixing mixed data types
    
def mixed2float(data):
    m = len(data)
    newData = np.zeros(m, dtype = np.float16)
    manyMarker = np.zeros(m, dtype = np.bool)
    
    for i in range(m):
        try:     # if number is already essentially in float format
            val = float(data[i])
            newData[i] = val
        except ValueError:     # some quirk with string
            # checks for large number commas comparisons for v small and large measurements
            string = re.sub('[ ,+><=]', '', data[i].lower())
            # checks for dashes to indicate ranges (e.g. 10-20)
            dashLoc = string.find('-')
            try:
                if string == 'none':     # sometimes 0 is coded as such
                    newData[i] = 0
                elif string == 'many':  # mark to impute largest value later
                    manyMarker[i] = True
                elif dashLoc != -1:      # check for dashes
                    val = (float(string[:dashLoc]) + float(string[dashLoc+1:]))/2
                    newData[i] = val
                else:           # can string be converted with just bad chars removed?
                    val = float(string)
                    newData[i] = val
            except ValueError:      # no hope. code as NaN (e.g. cancelled, unavilable, etc.)
                newData[i] = np.nan
    
    maxVal = np.max(newData[np.isfinite(newData)])
    newData[manyMarker] = maxVal
    return newData



def mixedCategoricalClean(data):
    """
    Makes string arrays more regular by removing comparisons, stuff in parentheses,
    units, capitalization, etc...
    """
    m = len(data)
    data = data.astype(np.str)
    select = ~(data == 'nan')
    for i in range(m):
        if select[i]:
            # reduces number of unique categories by creating more regularity in names
            string = re.sub('[><=,]', '', data[i].lower().replace('.0', '')).replace('..','.')
            data[i] = re.sub('\((.*?)\)','', string).replace('mg/dl', '').strip()
    return data
    

def mixedCategoricalUnique(data):
    """
    For string arrays, this provides names and counts of all unique items
    """
    data = mixedCategoricalClean(data)
    select = ~(data == 'nan')
    
    names, counts = np.unique(data[select], return_counts = True)
    sorter = np.argsort(counts)[::-1]
    names, counts = names[sorter], counts[sorter]
    return list(zip(list(names), list(counts)))
                
    



