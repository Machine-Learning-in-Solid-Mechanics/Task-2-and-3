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

    
def uniAxial(model, P, training_input):
    

    prediction = np.reshape(model.predict(training_input),(len(P),3,3))
    steps = np.arange(0,199)

    plt.figure(2, dpi=600)
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, prediction[0:199,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[0:199,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Uniaxial')
    plt.legend()
    plt.grid()
    plt.show()
        
def biAxial(model, P, training_input):
    

    prediction = np.reshape(model.predict(training_input),(len(P),3,3))
    steps = np.arange(0,199)

    plt.figure(2, dpi=600)
    #plt.scatter(xs_c[::10], ys_c[::10], c='green', label = 'calibration data')
    #plt.plot(steps, P[:,0,0], c='black', linestyle='--', label='function')
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, prediction[199:398,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[199:398,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Biaxial')
    plt.legend()
    plt.grid()
    plt.show()
    
def pureShear(model, P, training_input):
    

    prediction = np.reshape(model.predict(training_input),(len(P),3,3))
    steps = np.arange(0,250)

    plt.figure(2, dpi=600)
    #plt.scatter(xs_c[::10], ys_c[::10], c='green', label = 'calibration data')
    #plt.plot(steps, P[:,0,0], c='black', linestyle='--', label='function')
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, prediction[398:648,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[398:648,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Pure Shear')
    plt.legend()
    plt.grid()
    plt.show()
    
    
def test_data_biaxial(model, P):
    Data = np.genfromtxt('02_hyperelasticity_I/test/biax_test.txt')
    F, P0, W0 = np.split(Data,[9,18], axis =1)
    num_cases = len(F)
    F = np.reshape(F,[num_cases,3,3])
    P0 = np.reshape(P0,[num_cases,3,3])
    W0 = np.reshape(W0,[num_cases,1])
    
    x = tf.constant(F, dtype='float32')
    C = (tf.linalg.matrix_transpose(x,(0,2,1))@x).numpy()
    training_input = np.transpose(np.array((C[:,0,0], C[:,0,1],C[:,0,2],C[:,1,1],C[:,1,2],C[:,2,2])))
    prediction = np.reshape(model.predict(training_input),(len(C),3,3))
    steps = np.arange(0,len(C))
    
    plt.figure(2, dpi=600)
    #plt.scatter(xs_c[::10], ys_c[::10], c='green', label = 'calibration data')
    #plt.plot(steps, P[:,0,0], c='black', linestyle='--', label='function')
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, prediction[:,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P0[:,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Test Biaxial')
    plt.legend()
    plt.grid()
    plt.show()
    
def test_data_Mixed(model, P):
    Data = np.genfromtxt('02_hyperelasticity_I/test/mixed_test.txt')
    F, P0, W0 = np.split(Data,[9,18], axis =1)
    num_cases = len(F)
    F = np.reshape(F,[num_cases,3,3])
    P0 = np.reshape(P0,[num_cases,3,3])
    W0 = np.reshape(W0,[num_cases,1])
    
    x = tf.constant(F, dtype='float32')
    C = (tf.linalg.matrix_transpose(x,(0,2,1))@x).numpy()
    training_input = np.transpose(np.array((C[:,0,0], C[:,0,1],C[:,0,2],C[:,1,1],C[:,1,2],C[:,2,2])))
    prediction = np.reshape(model.predict(training_input),(len(C),3,3))
    steps = np.arange(0,len(C))
    
    plt.figure(2, dpi=600)
    #plt.scatter(xs_c[::10], ys_c[::10], c='green', label = 'calibration data')
    #plt.plot(steps, P[:,0,0], c='black', linestyle='--', label='function')
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, prediction[:,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P0[:,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Test Mixed')
    plt.legend()
    plt.grid()
    plt.show()    
  