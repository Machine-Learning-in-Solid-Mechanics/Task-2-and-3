# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:53:54 2022

@author: Mark Körbel
"""

import numpy as np
import tensorflow as tf

def function(random, G7):

    Data = np.genfromtxt('data/BCC_biaxial.txt')
    F0, P0, SE0, ERROR0 = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F0)
    F0 = np.reshape(F0,[num_cases,3,3])
    P0 = np.reshape(P0,[num_cases,3,3])
    SE0 = np.reshape(SE0,[num_cases,1])
    ERROR0 = np.reshape(ERROR0,[num_cases,1])
    
    Data = np.genfromtxt('data/BCC_planar.txt')
    F1, P1, SE1, ERROR1 = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F1)
    F1 = np.reshape(F1,[num_cases,3,3])
    P1 = np.reshape(P1,[num_cases,3,3])
    SE1 = np.reshape(SE1,[num_cases,1])
    ERROR1 = np.reshape(ERROR1,[num_cases,1])
    
    Data = np.genfromtxt('data/BCC_shear.txt')
    F2, P2, SE2, ERROR2 = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F2)
    F2 = np.reshape(F2,[num_cases,3,3])
    P2 = np.reshape(P2,[num_cases,3,3])
    SE2 = np.reshape(SE2,[num_cases,1])
    ERROR2 = np.reshape(ERROR2,[num_cases,1])
    
    Data = np.genfromtxt('data/BCC_test1.txt')
    F3, P3, SE3, ERROR3 = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F3)
    F3 = np.reshape(F3,[num_cases,3,3])
    P3 = np.reshape(P3,[num_cases,3,3])
    SE3 = np.reshape(SE3,[num_cases,1])
    ERROR3 = np.reshape(ERROR3,[num_cases,1])
    
    Data = np.genfromtxt('data/BCC_test2.txt')
    F4, P4, SE4, ERROR4 = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F4)
    F4 = np.reshape(F4,[num_cases,3,3])
    P4 = np.reshape(P4,[num_cases,3,3])
    SE4 = np.reshape(SE4,[num_cases,1])
    ERROR4 = np.reshape(ERROR4,[num_cases,1])
    
    Data = np.genfromtxt('data/BCC_test3.txt')
    F5, P5, SE5, ERROR5 = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F5)
    F5 = np.reshape(F5,[num_cases,3,3])
    P5 = np.reshape(P5,[num_cases,3,3])
    SE5 = np.reshape(SE5,[num_cases,1])
    ERROR5 = np.reshape(ERROR5,[num_cases,1])
    
    Data = np.genfromtxt('data/BCC_uniaxial.txt')
    F6, P6, SE6, ERROR6 = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F6)
    F6 = np.reshape(F6,[num_cases,3,3])
    P6 = np.reshape(P6,[num_cases,3,3])
    SE6 = np.reshape(SE6,[num_cases,1])
    ERROR6 = np.reshape(ERROR6,[num_cases,1])
    
    Data = np.genfromtxt('data/BCC_volumetric.txt')
    F7, P7, SE7, ERROR7 = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F7)
    F7 = np.reshape(F7,[num_cases,3,3])
    P7 = np.reshape(P7,[num_cases,3,3])
    SE7 = np.reshape(SE7,[num_cases,1])
    ERROR7 = np.reshape(ERROR7,[num_cases,1])
    
    '''
    uniaxial, biaxial, planar, shear, volumetric, test1, test2, test3
    '''
    #F = np.concatenate((F6,F0,F1,F2,F7), axis = 0)
    F = np.concatenate((F6,F0,F2,F7), axis = 0)
    F_det = tf.linalg.det(F)
    cof_F = tf.linalg.matrix_transpose(tf.linalg.inv(F),(0,2,1))@ tf.linalg.diag(tf.tile(tf.expand_dims(tf.linalg.det(F), axis = 1), [1, 3]))
    F_in = tf.concat([tf.reshape(F,[len(F),9]),tf.reshape(cof_F,[len(F),9]),tf.reshape(F_det,[len(F),1])],axis=-1)
    #P = np.concatenate((P6,P0,P1,P2,P7), axis = 0)
    P = np.concatenate((P6,P0,P2,P7), axis = 0)
    SE = np.concatenate((SE6,SE0,SE1,SE2,SE7), axis = 0)
    ERROR = np.concatenate((ERROR6,ERROR0,ERROR1,ERROR2,ERROR7), axis = 0)
        
    
    x = tf.constant(F, dtype='float32')
    
    with tf.GradientTape() as g:
        g.watch(x)
        
        C = tf.linalg.matrix_transpose(x,(0,2,1))@x
        I1 = tf.linalg.trace(C)
        J = tf.linalg.det(x)
        G_ti = tf.constant([[4,0,0],[0,0.5,0],[0,0,0.5]])
        I4 = tf.linalg.trace(C@G_ti)
        I3 = tf.linalg.det(C)
        cof_C = tf.linalg.diag(tf.tile(tf.expand_dims(I3, axis = 1), [1, 3]) ) @ tf.linalg.inv(C)
        I2 = tf.linalg.trace(cof_C)
        I5 = tf.linalg.trace(cof_C@G_ti)
        I7 = tf.math.multiply(C[:,0,0],C[:,0,0]) + tf.math.multiply(C[:,1,1],C[:,1,1]) + tf.math.multiply(C[:,2,2],C[:,2,2])
        I11 = tf.math.multiply(cof_C[:,0,0],cof_C[:,0,0]) + tf.math.multiply(cof_C[:,1,1],cof_C[:,1,1]) + tf.math.multiply(cof_C[:,2,2],cof_C[:,2,2])
        W_calc = 8*I1 + 10*tf.math.square(J) - 56*tf.math.log(J) + 0.2*(tf.math.square(I4) + tf.math.square(I5)) - 44 
    
    P_calc  = g.gradient(W_calc, x) 
    P = P/1000
    W_calc = W_calc/1000
    
    I = np.transpose(np.array((I1.numpy(), I2.numpy(), J.numpy(), -J.numpy(), I7.numpy(), I11.numpy())))
 
    return F, P, SE, ERROR, I, W_calc, P_calc, F_in.numpy()


