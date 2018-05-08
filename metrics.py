#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:31:03 2018

@author: aditya
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


"""
Functions for evaluating ITEs for survival data
Add logrank estimator down here when you find the file
"""


# kaplan-meier nonparametric estimator for survival curve
def KM_Estimator(Y, T, weighted = True, plot = False):
    index = np.argsort(T)
    Y, T = Y[index], T[index]
    times = np.unique(T)
    
    m = len(times)
    n = np.zeros(m)
    d = np.zeros(m)
    c = np.zeros(m)
    
    for i in range(m):
        select = T >= times[i]
        n[i] = np.sum(select)
        select = T == times[i]
        selected = Y[select]
        d[i] = np.sum(selected)
        c[i] = np.sum(1-selected)
    
    if weighted:
        w = (n-c)/n
    else:
        w = np.ones(len(n))
    S = np.cumprod(w*(1-d/n))
    
    if plot:
        plt.plot(np.concatenate([np.zeros(1), times]), np.concatenate((np.ones(1), S)))
    
    return times, S




# restricted mean survival time; tau is restriction time
def RMST(Y, T, tau = None):
    times, S = KM_Estimator(Y, T)
    if tau is None:
        tau = np.sort(T)[-1]
    else:
        select = times <= tau
        times, S = times[select], S[select]
    
    deltas = times[1:] - times[:-1]
    area = np.sum(S[:-1]*deltas) + times[0] + (tau-times[-1])*S[-1]
    return area
    
    

# plots those who fall in aligned-recommendation group for 4 strategies:
# treat all, treat no one, randomly treat, and targeted treat (determined by U)
# returns difference in area between treat intelligently and randomly as metric
def strategyGraph(U, Y, T, A, tau, bins = 10, plot = False, save = None):
    treated = A == 1    
    
    ##########################
    ##### CALCULATE RMST #####
    
    # Calculate Restricted Mean Survival Times
    treatmentPerformance = RMST(Y[treated], T[treated], tau)
    controlPerformance = RMST(Y[~treated], T[~treated], tau)
    
    # rate in recommended group at different decision boundaries for targeted treatment strategy
    binSize = 100/bins
    decisionBoundaries = list(np.percentile(U, np.arange(binSize, 100, binSize)))
    recPerformances = np.zeros(bins + 1, dtype = np.float32)
    for i, threshold, in enumerate(decisionBoundaries): # rec treat all -> rec treat no one
        treatSelector = U >= decisionBoundaries[i]
        recSelector = np.logical_or(np.logical_and(treatSelector, treated), np.logical_and(~treatSelector, ~treated))
        recPerformances[i+1] = RMST(Y[recSelector], T[recSelector], tau)
    recPerformances[0] = treatmentPerformance # left boundary
    recPerformances[-1] = controlPerformance # right boundary
    
    ############################
    ##### PLOT and RETURN ######

    x = np.arange(0, 100 + binSize, binSize)/100  # x-axis: [0,...,1]
    if plot or save is not None:            
        fig, ax = plt.subplots()
        
        # plot targeted strategy
        # x-axis horizontally flipped so that left -> right = control -> treat
        ax.plot(np.flip(x, axis = 0), recPerformances, 'g', linewidth = 2.5, label = 'Data-Driven Strategy')
        
        # plot other strategies as straight lines
        ax.plot([0, 1], [controlPerformance, treatmentPerformance], 'k', linewidth = 1.5, label = 'Random Strategy')
        ax.plot([0, 1], [treatmentPerformance, treatmentPerformance], 'm--', linewidth = 1.5, label = 'Treat-All Strategy')
        ax.plot([0, 1], [controlPerformance, controlPerformance], 'c--', linewidth = 1.5, label = 'Treat-Nobody Strategy')
        
        # make it look pretty and then show
        legend = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        for label in legend.get_texts():
            label.set_fontsize('large')
        ax.set_xlabel('Fraction of Population Recommended for Treatment', fontsize=18)
        ax.set_ylabel('Resticted Mean Survival Time (Years)', fontsize=18)
        ax.tick_params(labelsize = 14)
        if plot:
            plt.show()
        if save is not None:
            plt.savefig(save)
    
    # return metric
    randomArea = np.trapz([treatmentPerformance, controlPerformance], [0, 1])
    targetedArea = np.trapz(recPerformances, x)
    return targetedArea - randomArea





"""
Needs refactoring, renaming, and better comments
"""


## use this as default
def performance(IDs, U, Y, T, bins = 5, graph = True, label = None):
    #combine and sort data
    sort_index = np.argsort(U)[::-1]
    df = pd.DataFrame({'Uplift': U, 'target': Y, 'treatment': T}, index = IDs).iloc[sort_index,:]
    
    # organize slices
    m = len(df)
    bin_size = int(np.floor(m/bins))
    slices = [slice(0, bin_size*(i+1)) for i in range(bins-1)]
    slices += [slice(0,m)]
    
    data_binned = [df.iloc[locations,:] for locations in slices]
    Y_t = [sum(df['target'][df['treatment']==1]) for df in data_binned]
    Y_c = [sum(df['target'][df['treatment']==0])for df in data_binned]
    N_t = [sum(df['treatment'] == 1) for df in data_binned]
    N_c = [sum(df['treatment'] == 0) for df in data_binned]
    U = [100*(Y_t[i] - Y_c[i]*N_t[i]/N_c[i] + Y_t[i]*N_c[i]/N_t[i] - Y_c[i])/m for i in range(bins)]
    avg_effect = 100*(Y_t[-1]/N_t[-1] - Y_c[-1]/N_c[-1])
    
    X = [0] + [(b+1)*100/bins for b in list(range(bins))]
    Y = [0] + [U[i] for i in range(bins)]
    areas = [1/bins*(U[i]+U[i+1])/2 for i in range(bins-1)]
    AUUC = sum(areas) - avg_effect*1/2

    if graph:
        plt.plot(X, Y, label = label)
        plt.plot([0, 100], [0, avg_effect])
        plt.ylabel('% of Pop. Lives Saved')
        plt.xlabel('% of Pop. Targeted')
        plt.legend(loc = 'upper left')
    return(AUUC)




def bars(IDs, U, Y, T, bins = 5):
    #combine and sort data
    sort_index = np.argsort(U)[::-1]
    data = pd.DataFrame({'Uplift': U, 'target': Y, 'treatment': T}, index = IDs).iloc[sort_index,:]
    
    # organize slices
    m = len(data)
    bin_size = int(np.floor(m/bins))
    slices = [slice(bin_size*i, bin_size*(i+1)) for i in range(bins-1)]
    slices += [slice(bin_size*(bins-1),m)]
    
    data_binned = [data.iloc[locations,:] for locations in slices]
    Y_t = [sum(data['target'][data['treatment']==1]) for data in data_binned]
    Y_c = [sum(data['target'][data['treatment']==0])for data in data_binned]
    N_t = [sum(data['treatment'] == 1) for data in data_binned]
    N_c = [sum(data['treatment'] == 0) for data in data_binned]
    P_t = [Y_t[i]/N_t[i] for i in range(bins)]
    P_c = [Y_c[i]/N_c[i] for i in range(bins)]
    
    fig, ax = plt.subplots()
    width = 0.3
    rects1 = ax.bar(np.arange(bins), P_t, width = width, color = 'b')
    rects2 = ax.bar(np.arange(bins) + width, P_c, width = width, color = 'r')

    ax.set_ylabel('% of Pop. Lives Saved')
    ax.set_xlabel('% of Pop. Targeted')
    ax.set_ylim([0.85, 0.98])
    ax.legend((rects1[0], rects2[0]), ('Treated', 'Control'))
    plt.show()

    
