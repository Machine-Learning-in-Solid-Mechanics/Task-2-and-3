# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:55:38 2022

@author: jonat
"""

"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein, Henrik Hembrock, Jonathan Stollberg
         
08/2022
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

colors = np.array(['green', 'blue', 'red', 'brown', 'orange', 'purple', 'pink', 'gray', 'cyan'])
colors33 = colors.reshape([3,3])
    
def uniAxial(model, P_calc, training_input):
    
    prediction = np.reshape(model.predict(training_input)[1],(len(P_calc),3,3))
    steps = np.arange(0,50)

    plt.figure(1, dpi=600)
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, prediction[0:50,i,j], label='P'+str(i+1)+str(j+1),color=colors33[i,j])
            plt.plot(steps, P_calc[0:50,i,j],linestyle = 'dotted',color =colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Calibration1')
    plt.legend()
    plt.show()
        
def biAxial(model, P_calc, training_input):
    
    prediction = np.reshape(model.predict(training_input)[1],(len(P_calc),3,3))
    steps = np.arange(50,100)

    plt.figure(2, dpi=600)
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, prediction[50:100,i,j], label='P'+str(i+1)+str(j+1),color=colors33[i,j])
            plt.plot(steps, P_calc[50:100,i,j],linestyle = 'dotted',color=colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Calibration2')
    plt.legend()
    plt.show()
    
def test_70(model):
    
    F_test = np.genfromtxt('02_hyperelasticity_I/concentric/70.txt')
    F_test = np.reshape(F_test,[len(F_test),3,3])
    
    x = tf.constant(F_test, dtype='float32')
      
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
    
    P_test  = g.gradient(W_calc, x)
    
    training_input = tf.constant(F_test, dtype='float32')
    prediction = np.reshape(model.predict(training_input)[1],(len(C),3,3))
    steps = np.arange(0,50)
    
    plt.figure(3, dpi=600)
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, prediction[0:50,i,j], label='P'+str(i+1)+str(j+1),color=colors33[i,j])
            plt.plot(steps, P_test[0:50,i,j],linestyle = 'dotted',color=colors33[i,j],label='P'+str(i+1)+str(j+1))
        
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Test Data')
    plt.legend()
    plt.show()
    
def test_80(model):
    
    F_test = np.genfromtxt('02_hyperelasticity_I/concentric/80.txt')
    F_test = np.reshape(F_test,[len(F_test),3,3])
    
    x = tf.constant(F_test, dtype='float32')
    
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
    
    P_test  = g.gradient(W_calc, x)
    
    training_input = tf.constant(F_test, dtype='float32')
    prediction = np.reshape(model.predict(training_input)[1],(len(C),3,3))
    steps = np.arange(0,50)
    
    plt.figure(4, dpi=600)
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, prediction[:,i,j], label='P'+str(i+1)+str(j+1),color=colors33[i,j])
            plt.plot(steps, P_test[:,i,j],linestyle = 'dotted',color=colors33[i,j])
        
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Test Data')
    plt.legend()
    plt.show()
    
def test_90(model):
    
    F_test = np.genfromtxt('02_hyperelasticity_I/concentric/90.txt')
    F_test = np.reshape(F_test,[len(F_test),3,3])
    
    x = tf.constant(F_test, dtype='float32')
    
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
    
    P_test  = g.gradient(W_calc, x)
    
    training_input = tf.constant(F_test, dtype='float32')
    prediction = np.reshape(model.predict(training_input)[1],(len(C),3,3))
    steps = np.arange(0,50)
    
    plt.figure(5, dpi=600)
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, prediction[:,i,j], label='P'+str(i+1)+str(j+1),color=colors33[i,j])
            plt.plot(steps, P_test[:,i,j],linestyle = 'dotted',color=colors33[i,j])
        
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Test Data')
    plt.legend()
    plt.show()
    
def test_100(model):
    
    F_test = np.genfromtxt('02_hyperelasticity_I/concentric/100.txt')
    F_test = np.reshape(F_test,[len(F_test),3,3])
    
    x = tf.constant(F_test, dtype='float32')
    
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
    
    P_test  = g.gradient(W_calc, x)
    
    training_input = tf.constant(F_test, dtype='float32')
    prediction = np.reshape(model.predict(training_input)[1],(len(C),3,3))
    steps = np.arange(0,50)
    
    plt.figure(6, dpi=600)
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, prediction[:,i,j], label='P'+str(i+1)+str(j+1),color=colors33[i,j])
            plt.plot(steps, P_test[:,i,j],linestyle = 'dotted',color=colors33[i,j])
        
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Test Data')
    plt.legend()
    plt.show()
    
  