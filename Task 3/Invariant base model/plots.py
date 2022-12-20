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
from scipy.spatial.transform import Rotation as R

colors = np.array(['green', 'blue', 'red', 'brown', 'orange', 'purple', 'pink', 'gray', 'cyan'])

colors33 = colors.reshape([3,3])



def uniAxial(model):
    
    Data = np.genfromtxt('data/BCC_uniaxial.txt')
    F, P, SE, ERROR = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F)
    F = np.reshape(F,[num_cases,3,3])
    P = np.reshape(P,[num_cases,3,3])
    P = P/1000
    SE = np.reshape(SE,[num_cases,1])
    ERROR = np.reshape(ERROR,[num_cases,1])
    x = tf.constant(F, dtype='float32')
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
    W = 8*I1 + 10*tf.math.square(J) - 56*tf.math.log(J) + 0.2*(tf.math.square(I4) + tf.math.square(I5)) - 44 
    W = W/1000 
    W = W.numpy()
    
    W_mod, P_mod = model.predict(F)
    steps = np.arange(0,num_cases)
       
    plt.figure(2, dpi=600)
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, P_mod[:,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[:,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('Load steps')
    plt.ylabel('P')
    plt.title('Uniaxial')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(2, dpi=600)
    plt.plot(steps, W_mod, label='W', color = colors33[i,j])
    plt.plot(steps, W[0:199],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('Load steps')
    plt.ylabel('W')
    plt.title('Uniaxial')
    plt.legend()
    plt.grid()
    plt.show()


def biaxial(model):
    
    Data = np.genfromtxt('data/BCC_biaxial.txt')
    F, P, SE, ERROR = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F)
    F = np.reshape(F,[num_cases,3,3])
    P = np.reshape(P,[num_cases,3,3])
    P = P/1000
    SE = np.reshape(SE,[num_cases,1])
    ERROR = np.reshape(ERROR,[num_cases,1])
    x = tf.constant(F, dtype='float32')
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
    W = 8*I1 + 10*tf.math.square(J) - 56*tf.math.log(J) + 0.2*(tf.math.square(I4) + tf.math.square(I5)) - 44 
    W = W/1000 
    W = W.numpy()
    
    W_mod, P_mod = model.predict(F)
    steps = np.arange(0,num_cases)
       
    plt.figure(2, dpi=600)
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, P_mod[:,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[:,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('Load steps')
    plt.ylabel('P')
    plt.title('biaxial')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(2, dpi=600)
    plt.plot(steps, W_mod, label='W', color = colors33[i,j])
    plt.plot(steps, W[0:199],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('Load steps')
    plt.ylabel('W')
    plt.title('biaxial')
    plt.legend()
    plt.grid()
    plt.show()


def planar(model):
    
    Data = np.genfromtxt('data/BCC_planar.txt')
    F, P, SE, ERROR = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F)
    F = np.reshape(F,[num_cases,3,3])
    P = np.reshape(P,[num_cases,3,3])
    P = P/1000
    SE = np.reshape(SE,[num_cases,1])
    ERROR = np.reshape(ERROR,[num_cases,1])
    x = tf.constant(F, dtype='float32')
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
    W = 8*I1 + 10*tf.math.square(J) - 56*tf.math.log(J) + 0.2*(tf.math.square(I4) + tf.math.square(I5)) - 44 
    W = W/1000 
    W = W.numpy()
    
    W_mod, P_mod = model.predict(F)
    steps = np.arange(0,num_cases)
       
    plt.figure(2, dpi=600)
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, P_mod[:,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[:,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('Load steps')
    plt.ylabel('P')
    plt.title('planar')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(2, dpi=600)
    plt.plot(steps, W_mod, label='W', color = colors33[i,j])
    plt.plot(steps, W[0:199],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('Load steps')
    plt.ylabel('W')
    plt.title('planar')
    plt.legend()
    plt.grid()
    plt.show()


def shear(model):
    
    Data = np.genfromtxt('data/BCC_shear.txt')
    F, P, SE, ERROR = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F)
    F = np.reshape(F,[num_cases,3,3])
    P = np.reshape(P,[num_cases,3,3])
    P = P/1000
    SE = np.reshape(SE,[num_cases,1])
    ERROR = np.reshape(ERROR,[num_cases,1])
    x = tf.constant(F, dtype='float32')
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
    W = 8*I1 + 10*tf.math.square(J) - 56*tf.math.log(J) + 0.2*(tf.math.square(I4) + tf.math.square(I5)) - 44 
    W = W/1000 
    W = W.numpy()
    
    W_mod, P_mod = model.predict(F)
    steps = np.arange(0,num_cases)
       
    plt.figure(2, dpi=600)
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, P_mod[:,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[:,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('Load steps')
    plt.ylabel('P')
    plt.title('shear')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(2, dpi=600)
    plt.plot(steps, W_mod, label='W', color = colors33[i,j])
    plt.plot(steps, W[0:199],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('Load steps')
    plt.ylabel('W')
    plt.title('shear')
    plt.legend()
    plt.grid()
    plt.show()



def test1(model):
    
    Data = np.genfromtxt('data/BCC_test1.txt')
    F, P, SE, ERROR = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F)
    F = np.reshape(F,[num_cases,3,3])
    P = np.reshape(P,[num_cases,3,3])
    P = P/1000
    SE = np.reshape(SE,[num_cases,1])
    ERROR = np.reshape(ERROR,[num_cases,1])
    x = tf.constant(F, dtype='float32')
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
    W = 8*I1 + 10*tf.math.square(J) - 56*tf.math.log(J) + 0.2*(tf.math.square(I4) + tf.math.square(I5)) - 44 
    W = W/1000 
    W = W.numpy()
    
    W_mod, P_mod = model.predict(F)
    steps = np.arange(0,num_cases)
       
    plt.figure(2, dpi=600)
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, P_mod[:,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[:,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('Load steps')
    plt.ylabel('P')
    plt.title('test1')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(2, dpi=600)
    plt.plot(steps, W_mod, label='W', color = colors33[i,j])
    plt.plot(steps, W[0:199],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('Load steps')
    plt.ylabel('W')
    plt.title('test1')
    plt.legend()
    plt.grid()
    plt.show()



def test2(model):
    
    Data = np.genfromtxt('data/BCC_test2.txt')
    F, P, SE, ERROR = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F)
    F = np.reshape(F,[num_cases,3,3])
    P = np.reshape(P,[num_cases,3,3])
    P = P/1000
    SE = np.reshape(SE,[num_cases,1])
    ERROR = np.reshape(ERROR,[num_cases,1])
    x = tf.constant(F, dtype='float32')
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
    W = 8*I1 + 10*tf.math.square(J) - 56*tf.math.log(J) + 0.2*(tf.math.square(I4) + tf.math.square(I5)) - 44 
    W = W/1000 
    W = W.numpy()
    
    W_mod, P_mod = model.predict(F)
    steps = np.arange(0,num_cases)
       
    plt.figure(2, dpi=600)
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, P_mod[:,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[:,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('Load steps')
    plt.ylabel('P')
    plt.title('test2')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(2, dpi=600)
    plt.plot(steps, W_mod, label='W', color = colors33[i,j])
    plt.plot(steps, W[0:199],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('Load steps')
    plt.ylabel('W')
    plt.title('test2')
    plt.legend()
    plt.grid()
    plt.show()


def test3(model):
    
    Data = np.genfromtxt('data/BCC_test3.txt')
    F, P, SE, ERROR = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F)
    F = np.reshape(F,[num_cases,3,3])
    P = np.reshape(P,[num_cases,3,3])
    P = P/1000
    SE = np.reshape(SE,[num_cases,1])
    ERROR = np.reshape(ERROR,[num_cases,1])
    x = tf.constant(F, dtype='float32')
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
    W = 8*I1 + 10*tf.math.square(J) - 56*tf.math.log(J) + 0.2*(tf.math.square(I4) + tf.math.square(I5)) - 44 
    W = W/1000 
    W = W.numpy()
    
    W_mod, P_mod = model.predict(F)
    steps = np.arange(0,num_cases)
       
    plt.figure(2, dpi=600)
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, P_mod[:,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[:,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('Load steps')
    plt.ylabel('P')
    plt.title('test3')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(2, dpi=600)
    plt.plot(steps, W_mod, label='W', color = colors33[i,j])
    plt.plot(steps, W[0:199],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('Load steps')
    plt.ylabel('W')
    plt.title('test3')
    plt.legend()
    plt.grid()
    plt.show()


def volumetric(model):
    
    Data = np.genfromtxt('data/BCC_volumetric.txt')
    F, P, SE, ERROR = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F)
    F = np.reshape(F,[num_cases,3,3])
    P = np.reshape(P,[num_cases,3,3])
    P = P/1000
    SE = np.reshape(SE,[num_cases,1])
    ERROR = np.reshape(ERROR,[num_cases,1])
    x = tf.constant(F, dtype='float32')
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
    W = 8*I1 + 10*tf.math.square(J) - 56*tf.math.log(J) + 0.2*(tf.math.square(I4) + tf.math.square(I5)) - 44 
    W = W/1000 
    W = W.numpy()
    
    W_mod, P_mod = model.predict(F)
    steps = np.arange(0,num_cases)
       
    plt.figure(2, dpi=600)
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, P_mod[:,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[:,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('Load steps')
    plt.ylabel('P')
    plt.title('volumetric')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(2, dpi=600)
    plt.plot(steps, W_mod, label='W', color = colors33[i,j])
    plt.plot(steps, W[0:199],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('Load steps')
    plt.ylabel('W')
    plt.title('volumetric')
    plt.legend()
    plt.grid()
    plt.show()
