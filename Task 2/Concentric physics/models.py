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
        self.ls = [layers.Dense(nodes, "softplus")]
        for l in range(nlayers - 1):
            self.ls += [layers.Dense(nodes, "softplus")]
            
        # scalar-valued output function
        self.ls += [layers.Dense(9)]
            
    def __call__(self, x):     
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
        
    def call(self, xs):
        with tf.GradientTape() as tape:
            tape.watch(xs)
            ys = self.ls(xs)
        gs = tape.gradient(ys, xs)
        
        return ys, gs
    
def main(**kwargs):
    # define input shape
    xs = tf.keras.Input(shape=[6])
    # define which (custom) layers the model uses
    ys = _x_to_y(**kwargs)(xs)
    # connect input and output
    model = tf.keras.Model(inputs = [xs], outputs = [ys])
    # loss_weights = [0, 1]  # only gradient: [0,1], only output: [1,0]
    model.compile("adam", "mse", loss_weights=[1,1])
    return model