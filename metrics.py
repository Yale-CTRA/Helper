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


# normal kaplan-meier nonparametric estimator for survival curve
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
        #w = (n-c)/n
        w = getSurvivalStats(1-Y, T, times)
    else:
        w = np.ones(len(n))
    S = w*np.cumprod((1-d/n))
    
    if plot:
        plt.plot(np.concatenate([np.zeros(1), times]), np.concatenate((np.ones(1), S)))
    
    return times, S


def getSurvivalStats(Y, T, times):
    m = len(times)
    n, d = np.ones(m, dtype = np.int32), np.zeros(m, dtype = np.int32)
    for i in range(m):
        select = T >= times[i]
        n[i] = np.sum(select)
        select = T == times[i]
        selected = Y[select]
        d[i] = np.sum(selected)
    S = np.cumprod(1 - d/n)
    return S

## weighted estimator for difference of integration over 2 sample KM estimator
def WKM_Statistic(Y, T, A, scale = True):
    index = np.argsort(T)
    Y, T, A = Y[index], T[index], A[index]
    not_A = np.logical_not(A)
    
    Y_1, Y_0 = Y[A], Y[not_A]
    T_1, T_0 = T[A], T[not_A]
    times = np.unique(T)
    minLastT = np.min([T_1[-1], T_0[-1]])
    times = times[:np.where(times==minLastT)[0][0]]

    
    N_1, N_0 = len(Y_1), len(Y_0)
    N = N_1 + N_0
    p_1, p_0 = N_1/N, N_0/N
    
    C_1, C_0 = getSurvivalStats(1-Y_1, T_1, times), getSurvivalStats(1-Y_0, T_0, times)
    W = C_1*C_0/(p_1*C_1 + p_0*C_0)
    
    # get mu
    S_1, S_0 = getSurvivalStats(Y_1, T_1, times), getSurvivalStats(Y_0, T_0, times)
    deltas = times[1:] - times[:-1]
    U = np.sum(W[:-1]*(S_1[:-1] - S_0[:-1])*deltas)
    
#    # get variance
#    S = np.concatenate([np.array([1]), getSurvivalStats(Y, T, times)])
#    np.cumsum((W*S[:-1])[::-1])[::-1]
    if scale:
        return np.sqrt(N_1*N_0/N)*U
    else:
        return U
    
    
    
    
    




# restricted mean survival time; tau is restriction time
def RMST(Y, T, tau = None, weighted = False):
    times, S = KM_Estimator(Y, T, weighted = weighted)
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
#    treatmentPerformance = WKM_Statistic(Y, T, treated)
#    controlPerformance = - treatmentPerformance
    
    # rate in recommended group at different decision boundaries for targeted treatment strategy
    binSize = 100/bins
    decisionBoundaries = list(np.percentile(U, np.arange(binSize, 100, binSize)))
    recPerformances = np.zeros(bins + 1, dtype = np.float32)
    for i, threshold, in enumerate(decisionBoundaries): # rec treat all -> rec treat no one
        treatSelector = U >= decisionBoundaries[i]
        recSelector = np.logical_or(np.logical_and(treatSelector, treated), np.logical_and(~treatSelector, ~treated))
        #recPerformances[i+1] = WKM_Statistic(Y, T, recSelector)
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
    
    
    
class TimeEvaluator(object):
    def __init__(self, pIndex, Y, P, T, upper = 52, lower = 0, boundaries = None, 
                 n_bins = 100, modelName = ''):
        self.pIndex = pIndex
        self.Y = self.makeBool(Y)
        self.P = P
        self.T = T
        self.upper, self.lower = upper, lower
        self.n_bins = n_bins if boundaries is None else len(boundaries)-1
        self.modelName = modelName
        
        ### create usefulvars
        if boundaries is None:
            self.boundaries = self.makeBoundaries()
        else:
            self.boundaries = np.array(boundaries) if type(boundaries) is list else boundaries
        self.nEvents = self.countEvents()
        
        # record informaton
        self.lead = np.full((len(pIndex), self.n_bins+1), np.nan, dtype = np.float64)
        self.TPselect = np.zeros(self.lead.shape, dtype = np.bool)
        self.confusion = np.zeros((2, 2, self.n_bins+1), dtype = np.int64)
        self.record()
        self.medTPlead = np.full(self.n_bins+1, np.nan, dtype = np.float64)
        if self.n_bins > 0:
            for j in range(self.n_bins):
                self.medTPlead[j] = np.median(self.lead[self.TPselect[:,j],j])
        else:
            self.medTPlead[0] = np.median(self.lead[self.TPselect[:,0],0])
        
        
        # get statistics over confusion matrix
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
            self.TPR = self.confusion[1,1,:] / (self.confusion[1,1,:] + self.confusion[1,0,:])
            self.FNR = 1  - self.TPR
            self.FPR = self.confusion[0,1,:] / (self.confusion[0,0,:] + self.confusion[0,1,:])
            self.TNR = 1 - self.FPR
            self.PPV = self.confusion[1,1,:] / (self.confusion[1,1,:] + self.confusion[0,1,:])
            self.FDR = 1- self.PPV
            self.NPV = self.confusion[0,0,:] / (self.confusion[0,0,:] + self.confusion[1,0,:])
            self.FOR = 1 - self.NPV
            self.ACC = (self.confusion[0,0,:] + self.confusion[1,1,:])
            self.ACC = self.ACC/(self.ACC + self.confusion[0,1,:] + self.confusion[1,0,:])
        
        
        self.fullNames = ['true positive rate', 'true negative rate', 'false positive rate',
                          'false negative rate', 'positive predictive value', 
                          'negative predictive value', 'false discovery rate', 
                          'false omission rate', 'recall', 'precision', 'sensitivity',
                          'specificity', 'selectivity', 'miss rate', 'hit rate', 'fallout',
                          'accuracy']
        
        self.abbreviate = {'true positive rate': 'tpr', 'true negative rate':'tnr',
                            'false positive rate': 'fpr', 'false negative rate': 'fnr',
                            'positive predictive value': 'ppv', 'negative predictive value': 'npv',
                            'false discovery rate': 'fdr', 'false omission rate': 'for', 
                            'recall': 'tpr', 'precision': 'ppv', 'sensitivity': 'tpr',
                            'specificity': 'tnr', 'selectivity': 'tnr', 'miss rate': 'fnr', 
                            'hit rate': 'tpr', 'fallout': 'fpr', 'accuracy': 'acc'}
                                 
        self.map = {'tpr': self.TPR, 'tnr': self.TNR, 'fpr': self.FPR,'fnr': self.FNR,
                    'ppv': self.PPV, 'npv': self.NPV, 'fdr': self.FDR, 'for': self.FOR,
                    'acc': self.ACC, 'lead': self.medTPlead}
    
    def translate(self, string):
        string = string.lower()
        key = self.abbreviate[string.lower()] if string in self.fullNames else string
        return self.map[key]
    
    def __getitem__(self, string):
        return self.translate(string)
    
    def makeBool(self, Z):
        Z = Z if Z.dtype is np.dtype('bool') else Z==1 # convert to boolean if necessary
        return Z
    
    def makeBoundaries(self):
        EPSILON = 1e-7
        boundaries = np.percentile(self.P[self.Y], 100*np.arange(0,self.n_bins+1)/self.n_bins)
        boundaries[0], boundaries[-1] = -EPSILON, 1+EPSILON
        return boundaries
    
    def countEvents(self):
        counter = 0
        for i in range(len(self.pIndex)):
            start, stop, _ = self.pIndex[i,:]
            if np.any(self.Y[start:stop]):
                counter +=1
        return counter


    def record(self):
        # record where predictions cross decision boundaries using broadcasting
        crossings = self.P[np.newaxis,:] > self.boundaries[:,np.newaxis]
        # loop through encounters; add 1 to each confusion matrix per encounter
        for i in range(len(self.pIndex)):
            start, stop, length = self.pIndex[i,:]
            AKI = np.any(self.Y[start:stop])
            # restrict rows we examine for those who had AKI btw [upper,lower] hours pre-AKI
            stop, start = start + length - np.searchsorted(self.T[start:stop][::-1],
                                        [self.lower, self.upper]) if AKI else (stop, start)
            # classify crossings
            if stop - start > 0:
                positive = np.any(crossings[:,start:stop], axis = 1)
                where = np.argmax(crossings[:,start:stop], axis = 1)
                self.lead[i, positive] = self.T[start + where[positive]]
                if AKI:
                    self.TPselect[i,:] = positive
                    positive = positive.astype(np.int64)
                    self.confusion[1,1,:] += positive # True Positives (TP)
                    self.confusion[1,0,:] += 1 - positive # False Negatives (FN)
                else:
                    positive = positive.astype(np.int64)
                    self.confusion[0,1,:] += positive # False Positives (FP)
                    self.confusion[0,0,:] += 1 - positive # True Negatives (TN)
    
    
    def auc(self, plot = True):
        if plot:
            fig, ax = plt.subplots()
            ax.plot(self.FPR, self.TPR, 'b', linewidth = 1.5)
            ax.plot([0,1], [0,1], 'k', linewidth = 0.5, linestyle = '--')
            ax.set_xlabel('False Positive Rate', fontsize=18)
            ax.set_ylabel('True Positive Rate', fontsize=18)
            ax.tick_params(labelsize = 14)
            plt.show()
        return np.round(-np.trapz(self.TPR, x = self.FPR), 3) 
    
    def restrict(self, y, x, xrange):
        select = np.isfinite(y)
        y, x = y[select], x[select]
        if xrange is not None:
            select = np.logical_and(xrange[0] <= x,x <= xrange[1])
            y, x = y[select], x[select]
        if len(y) == 1:
            y = np.array([y[0], y[0]])
            x = np.array([0, 1]) if xrange is None else np.array(xrange)
        return y, x
    
    def plot(self, axis1, axis2 = None, xrange = None, compareto = None):
        fontsize = 24
        
        fig, ax1 = plt.subplots()
        y1a, x1a = self.restrict(self[axis1], self.boundaries, xrange = xrange)
        ax1.plot(x1a, y1a, 'b-', label = self.modelName, linewidth = 1.5)
        if compareto is not None:
            y1b, x1b = self.restrict(compareto[axis1], compareto.boundaries, xrange = xrange)
            ax1.plot(x1b, y1b, 'b--', label = compareto.modelName)
        ax1.legend()
        ax1.set_xlabel('decision boundaries', fontsize=fontsize)
        ax1.set_ylabel(axis1, color = 'b', fontsize=fontsize)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.tick_params('y', colors='b', labelsize = 18)
        
        if axis2 is not None:
            y2, x2 = self.restrict(self[axis2], self.boundaries, xrange = xrange)
            ax2 = ax1.twinx()
            ax2.plot(x2, y2, 'r-', linewidth = 1.5)
            if compareto is not None:
                y2b, x2b = self.restrict(compareto[axis2], compareto.boundaries, xrange = xrange)
                ax2.plot(x2b, y2b, 'r--')
            ax2.set_ylabel(axis2, color='r', fontsize=fontsize)
            ax2.tick_params('y', colors='r', labelsize = 18)

        plt.show()
        

    
