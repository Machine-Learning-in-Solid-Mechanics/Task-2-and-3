# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:53:54 2022

@author: arredond
"""

import numpy as np
import tensorflow as tf



def function():

    Data = np.genfromtxt('02_hyperelasticity_I/concentric/1.txt')
    F4 = Data
    for i in range(2,70):
        Data = np.genfromtxt('02_hyperelasticity_I/concentric/'+ str(i)+'.txt')
        F4 = np.concatenate((F4,Data), axis=0)    
    F4 = np.reshape(F4,[len(F4),3,3])
    
    x = tf.constant(F4, dtype='float32')
    
    
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
    
    return P_calc.numpy(), C.numpy()