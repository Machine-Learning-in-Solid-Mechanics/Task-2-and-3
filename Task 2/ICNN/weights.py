# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:45:40 2022

@author: arredond
"""

# Python3 implementation of the approach
import numpy as np


def weights(x):
    
    w0 = 0
    num_cases0 = 199
    for i in range(0,num_cases0):
        w0 += np.linalg.norm(x[i,:,:])
    w0 = 1/(w0/num_cases0)
    sw0 = np.ones(num_cases0,)*w0
    
    w1 = 0
    num_cases1 = 199
    for i in range(num_cases0,num_cases0 + num_cases1):
        w1 += np.linalg.norm(x[i,:,:])
    w1 = 1/(w1/num_cases1)
    sw1 = np.ones(num_cases1,)*w1
    
    w2 = 0
    num_cases2 = 250
    for i in range(num_cases0 + num_cases1,num_cases0+num_cases1+num_cases2):
        w2 += np.linalg.norm(x[i,:,:])
    w2 = 1/(w2/num_cases2)
    sw2 = np.ones(num_cases2,)*w2
    
    sw = np.concatenate((sw0,sw1,sw2), axis = 0)
    
    return sw
    
    
    
    

