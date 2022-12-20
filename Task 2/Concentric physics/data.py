"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein, Henrik Hembrock, Jonathan Stollberg
         
08/2022
"""
  
import numpy as np
from tensorflow.keras import layers

#%% data generation functions



def f2_data():
    """
    Generate data for `f2 = x**2 + 0.5*y**2`.
    """
    xs = np.linspace(-4, 4, 20)
    ys = np.linspace(-4, 4, 20)
    xs, ys = np.meshgrid(xs, ys)
    
    # Cut out the 4x4 grid in the middle for calibration data
    cut = np.concatenate([range(0,8), range(12,20)])
    cut = np.ix_(cut, cut)
    xs_c = xs[cut]
    ys_c = ys[cut]
    
    xs = xs.reshape((-1,1))
    ys = ys.reshape((-1,1))
    (zs, grad) = F2()(xs, ys)
    
    xs_c = xs_c.reshape((-1,1))
    ys_c = ys_c.reshape((-1,1))
    (zs_c, grad_c) = F2()(xs_c, ys_c)
    
    return xs, ys, zs, grad, xs_c, ys_c, zs_c, grad_c

#%% non-trainable layers

class F2(layers.Layer):
    """
    Non-trainable layer `f2 = x**2 + 0.5*y**2`.
    """
    def __call__(self, x, y):
        return x**2 + 0.5 + y**2, np.hstack([2*x, y])