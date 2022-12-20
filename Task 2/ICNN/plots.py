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
    
def uniAxial(model, P, W, training_input):
    

    P_gen = model.predict(training_input)[1]
    W_gen = model.predict(training_input)[0]
    steps = np.arange(0,199)

    plt.figure(2, dpi=600)
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, P_gen[0:199,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[0:199,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Uniaxial')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(2, dpi=600)
    plt.plot(steps, W_gen[0:199], label='W', color = colors33[i,j])
    plt.plot(steps, W[0:199],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('W')
    plt.title('Uniaxial')
    plt.legend()
    plt.grid()
    plt.show()
        
def biAxial(model, P, W, training_input):
    

    P_gen = model.predict(training_input)[1]
    W_gen = model.predict(training_input)[0]
    steps = np.arange(0,199)

    plt.figure(2, dpi=600)
    #plt.scatter(xs_c[::10], ys_c[::10], c='green', label = 'calibration data')
    #plt.plot(steps, P[:,0,0], c='black', linestyle='--', label='function')
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, P_gen[199:398,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[199:398,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Biaxial')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(2, dpi=600)
    plt.plot(steps, W_gen[199:398], color = colors33[i,j], label='W')
    plt.plot(steps, W[199:398],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('W')
    plt.title('Biaxial')
    plt.legend()
    plt.grid()
    plt.show()
    
def pureShear(model, P, W, training_input):
    

    P_gen = model.predict(training_input)[1]
    W_gen = model.predict(training_input)[0]
    steps = np.arange(0,250)

    plt.figure(2, dpi=600)
    #plt.scatter(xs_c[::10], ys_c[::10], c='green', label = 'calibration data')
    #plt.plot(steps, P[:,0,0], c='black', linestyle='--', label='function')
    for i in range(0,3):
        for j in range(0,3):
            #plt.plot(steps, P_gen[398:648,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P_gen[398:648,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[398:648,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Pure Shear')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(2, dpi=600)
    plt.plot(steps, W_gen[398:648], color = colors33[i,j], label='W')
    plt.plot(steps, W[398:648],linestyle = 'dotted')
    plt.xlabel('x')
    plt.ylabel('W')
    plt.title('Pure Shear')
    plt.legend()
    plt.grid()
    plt.show()
    
    
def test_data_biaxial(model):
    Data = np.genfromtxt('02_hyperelasticity_I/test/biax_test.txt')
    F, P0, W0 = np.split(Data,[9,18], axis =1)
    num_cases = len(F)
    F = np.reshape(F,[num_cases,3,3])
    P0 = np.reshape(P0,[num_cases,3,3])
    W0 = np.reshape(W0,[num_cases,1])
    
    training_input = tf.constant(F, dtype='float32')
    P_gen = model.predict(training_input)[1]
    W_gen = model.predict(training_input)[0]
    steps = np.arange(0,num_cases)
    
    plt.figure(2, dpi=600)
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, P_gen[:,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P0[:,i,j],linestyle = 'dotted', color = colors33[i,j])       
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Test Biaxial')
    plt.legend()
    plt.grid()
    plt.show()
    

    plt.figure(2, dpi=600)
    plt.plot(steps, W_gen[0:len(W_gen)], color = colors33[i,j], label='W')
    plt.plot(steps, W0[0:len(W_gen)],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('W')
    plt.title('Test Biaxial')
    plt.legend()
    plt.grid()
    plt.show()

def test_data_mixed(model):
    Data = np.genfromtxt('02_hyperelasticity_I/test/mixed_test.txt')
    F, P0, W0 = np.split(Data,[9,18], axis =1)
    num_cases = len(F)
    F = np.reshape(F,[num_cases,3,3])
    P0 = np.reshape(P0,[num_cases,3,3])
    W0 = np.reshape(W0,[num_cases,1])
    
    training_input = tf.constant(F, dtype='float32')
    P_gen = model.predict(training_input)[1]
    W_gen = model.predict(training_input)[0]
    steps = np.arange(0,num_cases)
    
    plt.figure(2, dpi=600)
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, P_gen[:,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P0[:,i,j],linestyle = 'dotted', color = colors33[i,j])       
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Test Mixed')
    plt.legend()
    plt.grid()
    plt.show()
    

    plt.figure(2, dpi=600)
    plt.plot(steps, W_gen[0:len(W_gen)], color = colors33[i,j], label='W')
    plt.plot(steps, W0[0:len(W_gen)],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('W')
    plt.title('Test Mixed')
    plt.legend()
    plt.grid()
    plt.show()