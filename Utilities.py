#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:58:29 2018

@author: aditya
"""

import pickle


"""
Wrappers for pickling.  Just makes saving/loading objects a little less verbose
"""
def save(obj, file):
    with open(file + '.pickle', 'wb') as output:
        pickle.dump(obj, output, -1)
def load(file):
    with open(file + '.pickle', 'rb') as input:
        obj = pickle.load(input)
        return obj
