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
    
P_calc, C = function()
training_input = np.transpose(np.array((C[:,0,0], C[:,0,1],C[:,0,2],C[:,1,1],C[:,1,2],C[:,2,2])))
training_output = np.transpose(np.array((P_calc[:,0,0], P_calc[:,0,1],P_calc[:,0,2],P_calc[:,1,0],P_calc[:,1,1],P_calc[:,1,2],P_calc[:,2,0],P_calc[:,2,1],P_calc[:,2,2])))
    

#%% load model

model = lm.main()

#%% import wiegths 

#sw = w.weights(P)

#%% model calibration


t1 = now()
tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)

h = model.fit(training_input, training_output, epochs=1500, verbose=2)
#h = model.fit(training_input, training_output, epochs=1500, verbose=2, sample_weight=sw)

print(f"It took {now() - t1} sec to calibrate the model.")



#%% result plots

pl.uniAxial(model, P_calc, training_input)
pl.biAxial(model, P_calc, training_input)
#pl.pureShear(model, P_calc, training_input)
pl.test_70(model)
pl.test_80(model)
pl.test_90(model)
pl.test_100(model)


