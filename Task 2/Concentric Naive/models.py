"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein, Henrik Hembrock, Jonathan Stollberg
         
08/2022
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
        self.ls = [layers.Dense(nodes, "softplus", kernel_constraint= non_neg())]
        for l in range(nlayers - 1):
            self.ls += [layers.Dense(nodes, "softplus", kernel_constraint= non_neg())]
            
        # scalar-valued output function
        self.ls += [layers.Dense(1, kernel_constraint= non_neg())]
            
    def __call__(self, Fs):   
        
        C = tf.transpose(Fs,(0,2,1))@Fs
        I1 = tf.linalg.trace(C)
        J = tf.linalg.det(Fs)
        G_ti = tf.constant([[4,0,0],[0,0.5,0],[0,0,0.5]])
        I4 = tf.linalg.trace(C@G_ti)
        I3 = tf.linalg.det(C)
        cof_C = tf.linalg.diag( tf.tile(tf.expand_dims(I3, axis = 1), [1, 3]) ) @ tf.linalg.inv(C)
        I5 = tf.linalg.trace(cof_C@G_ti)
        x = tf.stack([I1,J,-J,I4,I5], axis = 1)
        
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
    model = tf.keras.Model(inputs = [xs], outputs = [ys, gs])
    #ys = _x_to_y(**kwargs)(xs)
    # connect input and output
    #model = tf.keras.Model(inputs = [xs], outputs = [ys])
    # loss_weights =  only output: [1,0], only gradient: [0,1], output and gradient [1,1]
    model.compile("adam", "mse", loss_weights=[1,1])
    return model