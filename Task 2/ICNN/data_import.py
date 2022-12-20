# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:53:54 2022

@author: arredond
"""

import numpy as np
import tensorflow as tf



def function():

    Data = np.genfromtxt('02_hyperelasticity_I/calibration/biaxial.txt')
    F0, P0, W0 = np.split(Data,[9,18], axis =1)
    num_cases = len(F0)
    F0 = np.reshape(F0,[num_cases,3,3])
    P0 = np.reshape(P0,[num_cases,3,3])
    W0 = np.reshape(W0,[num_cases,1])
    
    Data = np.genfromtxt('02_hyperelasticity_I/calibration/pure_shear.txt') 
    F1, P1, W1 = np.split(Data,[9,18], axis =1)
    num_cases = len(F1)
    F1 = np.reshape(F1,[num_cases,3,3])
    P1 = np.reshape(P1,[num_cases,3,3])
    W1 = np.reshape(W1,[num_cases,1])
    
    Data = np.genfromtxt('02_hyperelasticity_I/calibration/uniaxial.txt')
    F2, P2, W2 = np.split(Data,[9,18], axis =1)
    num_cases = len(F2)
    F2 = np.reshape(F2,[num_cases,3,3])
    P2 = np.reshape(P2,[num_cases,3,3])
    W2 = np.reshape(W2,[num_cases,1])
    
    F = np.concatenate((F2,F0,F1), axis = 0)
    P = np.concatenate((P2,P0,P1), axis = 0)
    W = np.concatenate((W2,W0,W1), axis = 0)
    
    
    x = tf.constant(F, dtype='float32')
    
    
    with tf.GradientTape() as g:
        g.watch(x)
        
        C = tf.linalg.matrix_transpose(x,(0,2,1))@x
        I1 = tf.linalg.trace(C)
        J = tf.linalg.det(x)
        G_ti = tf.constant([[4,0,0],[0,0.5,0],[0,0,0.5]])
        I4 = tf.linalg.trace(C@G_ti)
        I3 = tf.linalg.det(C)
        cof_C = tf.linalg.diag( tf.tile(tf.expand_dims(I3, axis = 1), [1, 3]) ) @ tf.linalg.inv(C)
        I5 = tf.linalg.trace(cof_C@G_ti)
        W_calc = 8*I1 + 10*tf.math.square(J) - 56*tf.math.log(J) + 0.2*(tf.math.square(I4) + tf.math.square(I5)) - 44 
    
    P_calc  = g.gradient(W_calc, x) 
    
    I = np.transpose(np.array((I1.numpy(), J.numpy(), -J.numpy(), I4.numpy(), I5.numpy())))

    return P_calc.numpy(), C.numpy(), I, W_calc.numpy(), F