# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:22:48 2022

@author: koerbel
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.constraints import non_neg

class _x_to_y(layers.Layer):
    """
    Custom trainable layer for scalar output.
    """
    def __init__(self, 
                 nlayers=3, 
                 nodes=16):
        super(_x_to_y, self).__init__()
        
        # define hidden layers with activation functions
        #self.ls = [layers.Dense(nodes, "softplus", 
                                 #kernel_constraint=non_neg())]
        self.ls = [layers.Dense(nodes, "softplus")]
        for l in range(nlayers - 1):
            self.ls += [layers.Dense(nodes, "softplus", 
                                     kernel_constraint=non_neg())]
        # scalar-valued output function
        self.ls += [layers.Dense(1, kernel_constraint=non_neg())]
            
    def __call__(self, Fs): 
        
        C = tf.transpose(Fs,(0,2,1))@Fs    
        I1 = tf.linalg.trace(C)
        J = tf.linalg.det(Fs)
        G_ti = tf.constant([[4,0,0],[0,0.5,0],[0,0,0.5]])
        I4 = tf.linalg.trace(C@G_ti)
        I3 = tf.linalg.det(C)
        cof_C = tf.linalg.diag(tf.tile(tf.expand_dims(I3, axis = 1), [1, 3]) ) @ tf.linalg.inv(C)
        I2 = tf.linalg.trace(cof_C)
        I5 = tf.linalg.trace(cof_C@G_ti)
        I7 = tf.math.multiply(C[:,0,0],C[:,0,0]) + tf.math.multiply(C[:,1,1],C[:,1,1]) + tf.math.multiply(C[:,2,2],C[:,2,2])
        I11 = tf.math.multiply(cof_C[:,0,0],cof_C[:,0,0]) + tf.math.multiply(cof_C[:,1,1],cof_C[:,1,1]) + tf.math.multiply(cof_C[:,2,2],cof_C[:,2,2])
        
        x = tf.stack([I1,I2,J,-J,I7,I11], axis = 1)

        #x = tf.stack([I1], axis = 1)
        
        
        for l in self.ls:
            x = l(x)
        return x
 
class _y_to_dy(tf.keras.Model):
    """
    Neural network that computes scalar output and its gradient.
    """
    def __init__(self):
        super(_y_to_dy, self).__init__()
        self.ls = _x_to_y()
        
    def call(self, Fs):
        with tf.GradientTape() as tape:
            tape.watch(Fs)
            ys = self.ls(Fs)
        gs = tape.gradient(ys, Fs)
        
        return ys, gs
    
def main(**kwargs):
    # define input shape
    xs = tf.keras.Input(shape=[3,3])
    # define which (custom) layers the model uses
    ys, gs = _y_to_dy(**kwargs)(xs)
    # connect input and output
    model = tf.keras.Model(inputs = [xs], outputs = [ys, gs])
    # loss_weights = [0, 1]  # only gradient: [0,1], only output: [1,0]
    model.compile("adam", "mse", loss_weights=[0,1])
    return model