# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:45:40 2022

@author: arredond
"""

# Python3 implementation of the approach
import numpy as np


def weights(x):
    
    w0 = 0
    num_cases0 = 175
    for i in range(0,num_cases0):
        w0 += np.linalg.norm(x[i,:,:])
    w0 = 1/(w0/num_cases0)
    sw0 = np.ones(num_cases0)*w0
    
    w1 = 0
    num_cases1 = 135 + num_cases0
    for i in range(num_cases0,num_cases1):
        w1 += np.linalg.norm(x[i,:,:])
    w1 = 1/(w1/135)
    sw1 = np.ones(135)*w1
    
    w2 = 0
    num_cases2 = 175 + num_cases1
    for i in range(num_cases1,num_cases2):
        w2 += np.linalg.norm(x[i,:,:])
    w2 = 1/(w2/175)
    sw2 = np.ones(175)*w2
    
    w3 = 0
    num_cases3 = 101 + num_cases2
    for i in range(num_cases2,num_cases3):
        w3 += np.linalg.norm(x[i,:,:])
    w3 = 1/(w3/101)
    sw3 = np.ones(101)*w3
    
    w4 = 0
    num_cases4 = 169 + num_cases3
    for i in range(num_cases3,num_cases4):
        w4 += np.linalg.norm(x[i,:,:])
    w4 = 1/(w4/167)
    sw4 = np.ones(169)*w4
    
    sw = np.concatenate((sw0,sw1,sw2,sw3,sw4), axis = 0)

    
    return sw
    
    
    
    

