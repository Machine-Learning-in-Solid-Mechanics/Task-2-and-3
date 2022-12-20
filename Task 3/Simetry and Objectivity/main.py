"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein, Henrik Hembrock, Jonathan Stollberg
         
08/2022
"""

import tensorflow as tf
import numpy as np
import datetime
import models as lm
import weights as w
import plots as pl
from cubicgroup import cubicgroup
from matplotlib import pyplot as plt
from data_import import function

now = datetime.datetime.now

# set this to avoid conflicts with matplotlib
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

#%% load data
G7,random,random2=cubicgroup()    
F, P, SE, ERROR, I, W_calc, P_calc, F_in= function(random, G7)
training_input = tf.constant(F, dtype='float32')
training_output = [tf.constant(W_calc, dtype='float32'),tf.constant(P, dtype='float32')]
    
#%% load model

model = lm.main()

#%% import wiegths 

#sw = w.weights(P)

#%% model calibration

t1 = now()
tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)

#h = model.fit(training_input, training_output, epochs=1500, verbose=2)
h = model.fit(training_input, training_output, epochs=1500, verbose=2)#, sample_weight=np.concatenate((sw[0:175],sw[485:586])))

print(f"It took {now() - t1} sec to calibrate the model.")

plt.figure(1, dpi=600)
plt.semilogy(h.history['loss'], label='training loss')
plt.grid(which='both')
plt.xlabel('calibration epoch')
plt.ylabel('log$_{10}$ MSE')
plt.legend()
plt.show()

#%% result plots

pl.uniAxial(model, P, training_input,random,random2)
pl.biAxial(model, P, training_input, random, random2)
pl.planar(model, P, training_input, random, random2)
pl.shear(model, P, training_input, random, random2)
pl.volumetric(model, P, training_input, random, random2)
#pl.uniAxialW(model, W_calc, training_input)
pl.test1_data(model,random,random2)
pl.test2_data(model,random,random2)
pl.test3_data(model,random,random2)

