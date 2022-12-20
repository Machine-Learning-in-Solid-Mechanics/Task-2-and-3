"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein, Henrik Hembrock, Jonathan Stollberg
         
08/2022
"""

import numpy as np
import tensorflow as tf
import datetime
import models as lm
import weights as w
import plots as pl

from data_import import function


now = datetime.datetime.now

# set this to avoid conflicts with matplotlib
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

#%% load data
    
P, C, I, W, F = function()
#training_input = np.transpose(np.array((C[:,0,0], C[:,0,1],C[:,0,2],C[:,1,1],C[:,1,2],C[:,2,2]))) 
training_input = tf.constant(F, dtype='float32')  #np.transpose((np.array((F[:,0,0],F[:,0,1],F[:,0,2],F[:,1,0],F[:,1,1],F[:,1,2],F[:,2,0],F[:,2,1],F[:,2,2]))))
P1 = P #np.transpose((np.array((P[:,0,0], P[:,0,1],P[:,0,2],P[:,1,0],P[:,1,1],P[:,1,2],P[:,2,0],P[:,2,1],P[:,2,2]))))
training_output = [W[398:648],P1[398:648,:,:]]

#%% load model

model = lm.main()

#%% import wiegths 

sw = w.weights(P)

#%% model calibration


t1 = now()
tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
#h = model.fit(training_input, training_output, epochs=1500, verbose=2)
h = model.fit(training_input[398:648,:,:], training_output, epochs=5000, verbose=2, sample_weight=sw[398:648])

print(f"It took {now() - t1} sec to calibrate the model.")
np.append


#%% result plots

pl.uniAxial(model, P, W, training_input)
pl.biAxial(model, P, W, training_input)
pl.pureShear(model, P, W, training_input)
pl.test_data_biaxial(model)
pl.test_data_mixed(model)
