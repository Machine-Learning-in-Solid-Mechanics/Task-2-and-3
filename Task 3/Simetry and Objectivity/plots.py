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

    
def uniAxial(model, P, training_input,random,random2):
    
    prediction = model.predict(training_input)[1]
    steps = np.arange(0,175)
    #random2 = np.concatenate((np.reshape(np.identity(3),[1,3,3]),random),axis=0)

    plt.figure(2, dpi=600)
    #plt.scatter(xs_c[::10], ys_c[::10], c='green', label = 'calibration data')
    #plt.plot(steps, P[:,0,0], c='black', linestyle='--', label='function')
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, prediction[0:175,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[0:175,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Uniaxial')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(3, dpi=600)
    P_plot = prediction[0:175,:,:]
    P_plot2 = P[0:175,:,:]
    for j in range(1,len(random2)):
        n = 755*j
        P_plot = np.concatenate((P_plot,np.linalg.inv(random2[j,:,:])@prediction[n:n+175,:,:]), axis = 0)
        P_plot2 = np.concatenate((P_plot2,np.linalg.inv(random2[j,:,:])@P[n:n+175,:,:]), axis = 0)
    for i in range(0,len(random2)):
        n = 175*i
        plt.plot(steps, P_plot[n:n+175,0,0], label='P'+str(i))#, color = colors[i])
        #plt.plot(steps, np.max(P_plot[n:n+175,0,0]), label='P'+str(i))#, color = colors[i])
        #plt.plot(steps, np.min(P_plot[n:n+175,0,0]), label='P'+str(i))#, color = colors[i])
        plt.plot(steps, P_plot2[n:n+175,0,0],linestyle = 'dotted', color = colors[2])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Uniaxial_observer')
    #plt.legend()
    plt.grid()
    plt.show()
    
    P_plot_test = np.reshape(P_plot,[len(random2),175,3,3])
    P_plot_test_max0=[]
    P_plot_test_min0=[]
    P_plot_test_max1=[]
    P_plot_test_min1=[]
    P_plot_test_max2=[]
    P_plot_test_min2=[]
    P_plot_test_max3=[]
    P_plot_test_min3=[]
    P_plot_test_max4=[]
    P_plot_test_min4=[]
    P_plot_test_max5=[]
    P_plot_test_min5=[]
    for i in range(0,175):
        P_plot_test_max0.append(np.max(P_plot_test[:,i,0,0]))
        P_plot_test_min0.append(np.min(P_plot_test[:,i,0,0]))
        P_plot_test_max1.append(np.max(P_plot_test[:,i,1,1]))
        P_plot_test_min1.append(np.min(P_plot_test[:,i,1,1]))
        P_plot_test_max2.append(np.max(P_plot_test[:,i,2,2]))
        P_plot_test_min2.append(np.min(P_plot_test[:,i,2,2]))
        P_plot_test_max3.append(np.max(P_plot_test[:,i,0,1]))
        P_plot_test_min3.append(np.min(P_plot_test[:,i,0,1]))
        P_plot_test_max4.append(np.max(P_plot_test[:,i,0,2]))
        P_plot_test_min4.append(np.min(P_plot_test[:,i,0,2]))
        P_plot_test_max5.append(np.max(P_plot_test[:,i,1,2]))
        P_plot_test_min5.append(np.min(P_plot_test[:,i,1,2]))
    plt.figure(4, dpi=600)
    plt.fill_between(steps,P_plot_test_max0,P_plot_test_min0, alpha = 0.5, color = colors[4], label='P11')
    plt.fill_between(steps,P_plot_test_max1,P_plot_test_min1, alpha = 0.5, color = colors[5], label='P22')
    plt.fill_between(steps,P_plot_test_max2,P_plot_test_min2, alpha = 0.5, color = colors[6], label='P33')
    plt.fill_between(steps,P_plot_test_max3,P_plot_test_min3, alpha = 0.5, color = colors[7], label='P12')
    plt.fill_between(steps,P_plot_test_max4,P_plot_test_min4, alpha = 0.5, color = colors[8], label='P13')
    plt.fill_between(steps,P_plot_test_max5,P_plot_test_min4, alpha = 0.5, color = 'green', label='P23')
    plt.plot(steps, P_plot2[0:175,0,0],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:175,1,1],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:175,2,2],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:175,0,1],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:175,0,2],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:175,1,2],linestyle = 'dotted', color = 'black')
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Uniaxial_Fill')
    plt.legend()
    plt.grid()
    plt.show()
        
def biAxial(model, P, training_input, random, random2):
    
    prediction = model.predict(training_input)[1]
    steps = np.arange(0,135)

    plt.figure(5, dpi=600)
    #plt.scatter(xs_c[::10], ys_c[::10], c='green', label = 'calibration data')
    #plt.plot(steps, P[:,0,0], c='black', linestyle='--', label='function')
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, prediction[175:310,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[175:310,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Biaxial')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(6, dpi=600)
    P_plot = prediction[175:310,:,:]
    P_plot2 = P[175:310,:,:]
    for j in range(1,len(random2)):
        n = 755*j+175
        P_plot = np.concatenate((P_plot,np.linalg.inv(random2[j,:,:])@prediction[n:n+135,:,:]), axis = 0)
        P_plot2 = np.concatenate((P_plot2,np.linalg.inv(random2[j,:,:])@P[n:n+135,:,:]), axis = 0)
    for i in range(0,len(random2)):
        n = 135*i
        plt.plot(steps, P_plot[n:n+135,0,0], label='P'+str(i))#, color = colors[i])
        #plt.plot(steps, np.max(P_plot[n:n+175,0,0]), label='P'+str(i))#, color = colors[i])
        #plt.plot(steps, np.min(P_plot[n:n+175,0,0]), label='P'+str(i))#, color = colors[i])
        plt.plot(steps, P_plot2[n:n+135,0,0],linestyle = 'dotted', color = colors[2])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Biiaxial_observer')
    #plt.legend()
    plt.grid()
    plt.show()
    
    P_plot_test = np.reshape(P_plot,[len(random2),135,3,3])
    P_plot_test_max0=[]
    P_plot_test_min0=[]
    P_plot_test_max1=[]
    P_plot_test_min1=[]
    P_plot_test_max2=[]
    P_plot_test_min2=[]
    P_plot_test_max3=[]
    P_plot_test_min3=[]
    P_plot_test_max4=[]
    P_plot_test_min4=[]
    P_plot_test_max5=[]
    P_plot_test_min5=[]
    for i in range(0,135):
        P_plot_test_max0.append(np.max(P_plot_test[:,i,0,0]))
        P_plot_test_min0.append(np.min(P_plot_test[:,i,0,0]))
        P_plot_test_max1.append(np.max(P_plot_test[:,i,1,1]))
        P_plot_test_min1.append(np.min(P_plot_test[:,i,1,1]))
        P_plot_test_max2.append(np.max(P_plot_test[:,i,2,2]))
        P_plot_test_min2.append(np.min(P_plot_test[:,i,2,2]))
        P_plot_test_max3.append(np.max(P_plot_test[:,i,0,1]))
        P_plot_test_min3.append(np.min(P_plot_test[:,i,0,1]))
        P_plot_test_max4.append(np.max(P_plot_test[:,i,0,2]))
        P_plot_test_min4.append(np.min(P_plot_test[:,i,0,2]))
        P_plot_test_max5.append(np.max(P_plot_test[:,i,1,2]))
        P_plot_test_min5.append(np.min(P_plot_test[:,i,1,2]))
    plt.figure(7, dpi=600)
    plt.fill_between(steps,P_plot_test_max0,P_plot_test_min0, alpha = 0.5, color = colors[4], label='P11')
    plt.fill_between(steps,P_plot_test_max1,P_plot_test_min1, alpha = 0.5, color = colors[5], label='P22')
    plt.fill_between(steps,P_plot_test_max2,P_plot_test_min2, alpha = 0.5, color = colors[6], label='P33')
    plt.fill_between(steps,P_plot_test_max3,P_plot_test_min3, alpha = 0.5, color = colors[7], label='P12')
    plt.fill_between(steps,P_plot_test_max4,P_plot_test_min4, alpha = 0.5, color = colors[8], label='P13')
    plt.fill_between(steps,P_plot_test_max5,P_plot_test_min4, alpha = 0.5, color = 'green', label='P23')
    plt.plot(steps, P_plot2[0:135,0,0],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:135,1,1],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:135,2,2],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:135,0,1],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:135,0,2],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:135,1,2],linestyle = 'dotted', color = 'black')
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Biaxial_Fill')
    plt.legend()
    plt.grid()
    plt.show()
    
def planar(model, P, training_input, random, random2):
    
    prediction = model.predict(training_input)[1]
    steps = np.arange(0,175)

    plt.figure(8, dpi=600)
    #plt.scatter(xs_c[::10], ys_c[::10], c='green', label = 'calibration data')
    #plt.plot(steps, P[:,0,0], c='black', linestyle='--', label='function')
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, prediction[310:485,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[310:485,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Planar')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(9, dpi=600)
    P_plot = prediction[310:485,:,:]
    P_plot2 = P[310:485,:,:]
    for j in range(1,len(random2)):
        n = 755*j+310
        P_plot = np.concatenate((P_plot,np.linalg.inv(random2[j,:,:])@prediction[n:n+175,:,:]), axis = 0)
        P_plot2 = np.concatenate((P_plot2,np.linalg.inv(random2[j,:,:])@P[n:n+175,:,:]), axis = 0)
    for i in range(0,len(random2)):
        n = 175*i
        plt.plot(steps, P_plot[n:n+175,0,0], label='P'+str(i))#, color = colors[i])
        #plt.plot(steps, np.max(P_plot[n:n+175,0,0]), label='P'+str(i))#, color = colors[i])
        #plt.plot(steps, np.min(P_plot[n:n+175,0,0]), label='P'+str(i))#, color = colors[i])
        plt.plot(steps, P_plot2[n:n+175,0,0],linestyle = 'dotted', color = colors[2])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Planar_observer')
    #plt.legend()
    plt.grid()
    plt.show()
    
    P_plot_test = np.reshape(P_plot,[len(random2),175,3,3])
    P_plot_test_max0=[]
    P_plot_test_min0=[]
    P_plot_test_max1=[]
    P_plot_test_min1=[]
    P_plot_test_max2=[]
    P_plot_test_min2=[]
    P_plot_test_max3=[]
    P_plot_test_min3=[]
    P_plot_test_max4=[]
    P_plot_test_min4=[]
    P_plot_test_max5=[]
    P_plot_test_min5=[]
    for i in range(0,175):
        P_plot_test_max0.append(np.max(P_plot_test[:,i,0,0]))
        P_plot_test_min0.append(np.min(P_plot_test[:,i,0,0]))
        P_plot_test_max1.append(np.max(P_plot_test[:,i,1,1]))
        P_plot_test_min1.append(np.min(P_plot_test[:,i,1,1]))
        P_plot_test_max2.append(np.max(P_plot_test[:,i,2,2]))
        P_plot_test_min2.append(np.min(P_plot_test[:,i,2,2]))
        P_plot_test_max3.append(np.max(P_plot_test[:,i,0,1]))
        P_plot_test_min3.append(np.min(P_plot_test[:,i,0,1]))
        P_plot_test_max4.append(np.max(P_plot_test[:,i,0,2]))
        P_plot_test_min4.append(np.min(P_plot_test[:,i,0,2]))
        P_plot_test_max5.append(np.max(P_plot_test[:,i,1,2]))
        P_plot_test_min5.append(np.min(P_plot_test[:,i,1,2]))
    plt.figure(10, dpi=600)
    plt.fill_between(steps,P_plot_test_max0,P_plot_test_min0, alpha = 0.5, color = colors[4], label='P11')
    plt.fill_between(steps,P_plot_test_max1,P_plot_test_min1, alpha = 0.5, color = colors[5], label='P22')
    plt.fill_between(steps,P_plot_test_max2,P_plot_test_min2, alpha = 0.5, color = colors[6], label='P33')
    plt.fill_between(steps,P_plot_test_max3,P_plot_test_min3, alpha = 0.5, color = colors[7], label='P12')
    plt.fill_between(steps,P_plot_test_max4,P_plot_test_min4, alpha = 0.5, color = colors[8], label='P13')
    plt.fill_between(steps,P_plot_test_max5,P_plot_test_min4, alpha = 0.5, color = 'green', label='P23')
    plt.plot(steps, P_plot2[0:175,0,0],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:175,1,1],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:175,2,2],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:175,0,1],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:175,0,2],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:175,1,2],linestyle = 'dotted', color = 'black')
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Planar_Fill')
    plt.legend()
    plt.grid()
    plt.show()

    
def shear(model, P, training_input, random, random2):
    
    prediction = model.predict(training_input)[1]
    steps = np.arange(0,101)

    plt.figure(11, dpi=600)
    #plt.scatter(xs_c[::10], ys_c[::10], c='green', label = 'calibration data')
    #plt.plot(steps, P[:,0,0], c='black', linestyle='--', label='function')
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, prediction[485:586,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[485:586,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Shear')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(12, dpi=600)
    P_plot = prediction[485:586,:,:]
    P_plot2 = P[485:586,:,:]
    for j in range(1,len(random2)):
        n = 755*j+485
        P_plot = np.concatenate((P_plot,np.linalg.inv(random2[j,:,:])@prediction[n:n+101,:,:]), axis = 0)
        P_plot2 = np.concatenate((P_plot2,np.linalg.inv(random2[j,:,:])@P[n:n+101,:,:]), axis = 0)
    for i in range(0,len(random2)):
        n = 101*i
        plt.plot(steps, P_plot[n:n+101,0,0], label='P'+str(i))#, color = colors[i])
        #plt.plot(steps, np.max(P_plot[n:n+175,0,0]), label='P'+str(i))#, color = colors[i])
        #plt.plot(steps, np.min(P_plot[n:n+175,0,0]), label='P'+str(i))#, color = colors[i])
        plt.plot(steps, P_plot2[n:n+101,0,0],linestyle = 'dotted', color = colors[2])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Shear_observer')
    #plt.legend()
    plt.grid()
    plt.show()
    
    P_plot_test = np.reshape(P_plot,[len(random2),101,3,3])
    P_plot_test_max0=[]
    P_plot_test_min0=[]
    P_plot_test_max1=[]
    P_plot_test_min1=[]
    P_plot_test_max2=[]
    P_plot_test_min2=[]
    P_plot_test_max3=[]
    P_plot_test_min3=[]
    P_plot_test_max4=[]
    P_plot_test_min4=[]
    P_plot_test_max5=[]
    P_plot_test_min5=[]
    for i in range(0,101):
        P_plot_test_max0.append(np.max(P_plot_test[:,i,0,0]))
        P_plot_test_min0.append(np.min(P_plot_test[:,i,0,0]))
        P_plot_test_max1.append(np.max(P_plot_test[:,i,1,1]))
        P_plot_test_min1.append(np.min(P_plot_test[:,i,1,1]))
        P_plot_test_max2.append(np.max(P_plot_test[:,i,2,2]))
        P_plot_test_min2.append(np.min(P_plot_test[:,i,2,2]))
        P_plot_test_max3.append(np.max(P_plot_test[:,i,0,1]))
        P_plot_test_min3.append(np.min(P_plot_test[:,i,0,1]))
        P_plot_test_max4.append(np.max(P_plot_test[:,i,0,2]))
        P_plot_test_min4.append(np.min(P_plot_test[:,i,0,2]))
        P_plot_test_max5.append(np.max(P_plot_test[:,i,1,2]))
        P_plot_test_min5.append(np.min(P_plot_test[:,i,1,2]))
    plt.figure(13, dpi=600)
    plt.fill_between(steps,P_plot_test_max0,P_plot_test_min0, alpha = 0.5, color = colors[4], label='P11')
    plt.fill_between(steps,P_plot_test_max1,P_plot_test_min1, alpha = 0.5, color = colors[5], label='P22')
    plt.fill_between(steps,P_plot_test_max2,P_plot_test_min2, alpha = 0.5, color = colors[6], label='P33')
    plt.fill_between(steps,P_plot_test_max3,P_plot_test_min3, alpha = 0.5, color = colors[7], label='P12')
    plt.fill_between(steps,P_plot_test_max4,P_plot_test_min4, alpha = 0.5, color = colors[8], label='P13')
    plt.fill_between(steps,P_plot_test_max5,P_plot_test_min4, alpha = 0.5, color = 'green', label='P23')
    plt.plot(steps, P_plot2[0:101,0,0],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:101,1,1],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:101,2,2],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:101,0,1],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:101,0,2],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:101,1,2],linestyle = 'dotted', color = 'black')
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Shear_Fill')
    plt.legend()
    plt.grid()
    plt.show()
    
def volumetric(model, P, training_input, random, random2):
    
    prediction = model.predict(training_input)[1]
    steps = np.arange(0,169)

    plt.figure(14, dpi=600)
    #plt.scatter(xs_c[::10], ys_c[::10], c='green', label = 'calibration data')
    #plt.plot(steps, P[:,0,0], c='black', linestyle='--', label='function')
    for i in range(0,3):
        for j in range(0,3):
            plt.plot(steps, prediction[586:755,i,j], color = colors33[i,j], label='P'+str(i+1)+str(j+1))
            plt.plot(steps, P[586:755,i,j],linestyle = 'dotted', color = colors33[i,j])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Volumetric')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(15, dpi=600)
    P_plot = prediction[586:755,:,:]
    P_plot2 = P[586:755,:,:]
    for j in range(1,len(random2)):
        n = 755*j+586
        P_plot = np.concatenate((P_plot,np.linalg.inv(random2[j,:,:])@prediction[n:n+169,:,:]), axis = 0)
        P_plot2 = np.concatenate((P_plot2,np.linalg.inv(random2[j,:,:])@P[n:n+169,:,:]), axis = 0)
    for i in range(0,len(random2)):
        n = 169*i
        plt.plot(steps, P_plot[n:n+169,0,0], label='P'+str(i))#, color = colors[i])
        #plt.plot(steps, np.max(P_plot[n:n+175,0,0]), label='P'+str(i))#, color = colors[i])
        #plt.plot(steps, np.min(P_plot[n:n+175,0,0]), label='P'+str(i))#, color = colors[i])
        plt.plot(steps, P_plot2[n:n+169,0,0],linestyle = 'dotted', color = colors[2])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Volumetric_observer')
    #plt.legend()
    plt.grid()
    plt.show()
    
    P_plot_test = np.reshape(P_plot,[len(random2),169,3,3])
    P_plot_test_max0=[]
    P_plot_test_min0=[]
    P_plot_test_max1=[]
    P_plot_test_min1=[]
    P_plot_test_max2=[]
    P_plot_test_min2=[]
    P_plot_test_max3=[]
    P_plot_test_min3=[]
    P_plot_test_max4=[]
    P_plot_test_min4=[]
    P_plot_test_max5=[]
    P_plot_test_min5=[]
    for i in range(0,169):
        P_plot_test_max0.append(np.max(P_plot_test[:,i,0,0]))
        P_plot_test_min0.append(np.min(P_plot_test[:,i,0,0]))
        P_plot_test_max1.append(np.max(P_plot_test[:,i,1,1]))
        P_plot_test_min1.append(np.min(P_plot_test[:,i,1,1]))
        P_plot_test_max2.append(np.max(P_plot_test[:,i,2,2]))
        P_plot_test_min2.append(np.min(P_plot_test[:,i,2,2]))
        P_plot_test_max3.append(np.max(P_plot_test[:,i,0,1]))
        P_plot_test_min3.append(np.min(P_plot_test[:,i,0,1]))
        P_plot_test_max4.append(np.max(P_plot_test[:,i,0,2]))
        P_plot_test_min4.append(np.min(P_plot_test[:,i,0,2]))
        P_plot_test_max5.append(np.max(P_plot_test[:,i,1,2]))
        P_plot_test_min5.append(np.min(P_plot_test[:,i,1,2]))
    plt.figure(16, dpi=600)
    plt.fill_between(steps,P_plot_test_max0,P_plot_test_min0, alpha = 0.5, color = colors[4], label='P11')
    plt.fill_between(steps,P_plot_test_max1,P_plot_test_min1, alpha = 0.5, color = colors[5], label='P22')
    plt.fill_between(steps,P_plot_test_max2,P_plot_test_min2, alpha = 0.5, color = colors[6], label='P33')
    plt.fill_between(steps,P_plot_test_max3,P_plot_test_min3, alpha = 0.5, color = colors[7], label='P12')
    plt.fill_between(steps,P_plot_test_max4,P_plot_test_min4, alpha = 0.5, color = colors[8], label='P13')
    plt.fill_between(steps,P_plot_test_max5,P_plot_test_min4, alpha = 0.5, color = 'green', label='P23')
    plt.plot(steps, P_plot2[0:169,0,0],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:169,1,1],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:169,2,2],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:169,0,1],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:169,0,2],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2[0:169,1,2],linestyle = 'dotted', color = 'black')
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Volumetric_Fill')
    plt.legend()
    plt.grid()
    plt.show()

def uniAxialW(model, W_calc, training_input):
    
    prediction = model.predict(training_input)[0]
    steps = np.arange(0,175)

    plt.figure(17, dpi=600)
    #plt.scatter(xs_c[::10], ys_c[::10], c='green', label = 'calibration data')
    #plt.plot(steps, P[:,0,0], c='black', linestyle='--', label='function')
    plt.plot(steps, prediction[0:175], color = 'blue', label='W')
    plt.plot(steps, W_calc[0:175],linestyle = 'dotted', color = 'blue')
    plt.xlabel('x')
    plt.ylabel('W')
    plt.title('Uniaxial_W')
    plt.legend()
    plt.grid()
    plt.show()
        

def test1_data(model, random, random2):
    Data = np.genfromtxt('data/BCC_test1.txt')
    F_test1, P_test1, SE_test1, ERROR_test1 = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F_test1)
    F_test1 = np.reshape(F_test1,[num_cases,3,3])
    P_test1 = np.reshape(P_test1,[num_cases,3,3])
    
    rand = np.random.rand(100,3)
    random_test = R.from_euler('XYZ',rand).as_matrix()
    random_test2 = np.concatenate((np.reshape(np.identity(3),[1,3,3]),random_test),axis=0)
    
    F_rot1 = []
    for i in range(0,len(random_test)):
        F_rot1.append(random_test[i,:,:]@F_test1)
        
    for l in range (0, len(F_rot1)):
        F_test1 = np.concatenate((F_test1,F_rot1[l]),axis=0)
    
    training_input_test1 = tf.constant(F_test1, dtype='float32')
    prediction = model.predict(training_input_test1)[1]
    steps = np.arange(0,145)
    
    plt.figure(18, dpi=600)
    P_plot_1 = prediction[0:145,:,:]
    P_plot2_1 = P_test1[0:145,:,:]
    for j in range(1,len(random_test2)):
        n = 145*j
        P_plot_1 = np.concatenate((P_plot_1,np.linalg.inv(random_test2[j,:,:])@prediction[n:n+145,:,:]), axis = 0)
        P_plot2_1 = np.concatenate((P_plot2_1,np.linalg.inv(random_test2[j,:,:])@P_test1[n:n+145,:,:]), axis = 0)
    P_plot2_1 = P_test1/1000
    for i in range(0,len(random_test2)):
        n = 145*i
        plt.plot(steps, P_plot_1[n:n+145,0,0], label='P'+str(i))#, color = colors[i])
        #plt.plot(steps, np.max(P_plot[n:n+175,0,0]), label='P'+str(i))#, color = colors[i])
        #plt.plot(steps, np.min(P_plot[n:n+175,0,0]), label='P'+str(i))#, color = colors[i])
        plt.plot(steps, P_plot2_1[0:145,0,0],linestyle = 'dotted', color = colors[2])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('test1_observer')
    #plt.legend()
    plt.grid()
    plt.show()
    
    P_plot_test = np.reshape(P_plot_1,[len(random_test2),145,3,3])
    P_plot_test_max0=[]
    P_plot_test_min0=[]
    P_plot_test_max1=[]
    P_plot_test_min1=[]
    P_plot_test_max2=[]
    P_plot_test_min2=[]
    P_plot_test_max3=[]
    P_plot_test_min3=[]
    P_plot_test_max4=[]
    P_plot_test_min4=[]
    P_plot_test_max5=[]
    P_plot_test_min5=[]
    for i in range(0,145):
        P_plot_test_max0.append(np.max(P_plot_test[:,i,0,0]))
        P_plot_test_min0.append(np.min(P_plot_test[:,i,0,0]))
        P_plot_test_max1.append(np.max(P_plot_test[:,i,1,1]))
        P_plot_test_min1.append(np.min(P_plot_test[:,i,1,1]))
        P_plot_test_max2.append(np.max(P_plot_test[:,i,2,2]))
        P_plot_test_min2.append(np.min(P_plot_test[:,i,2,2]))
        P_plot_test_max3.append(np.max(P_plot_test[:,i,0,1]))
        P_plot_test_min3.append(np.min(P_plot_test[:,i,0,1]))
        P_plot_test_max4.append(np.max(P_plot_test[:,i,0,2]))
        P_plot_test_min4.append(np.min(P_plot_test[:,i,0,2]))
        P_plot_test_max5.append(np.max(P_plot_test[:,i,1,2]))
        P_plot_test_min5.append(np.min(P_plot_test[:,i,1,2]))
    plt.figure(19, dpi=600)
    plt.fill_between(steps,P_plot_test_max0,P_plot_test_min0, alpha = 0.5, color = colors[4], label='P11')
    plt.fill_between(steps,P_plot_test_max1,P_plot_test_min1, alpha = 0.5, color = colors[5], label='P22')
    plt.fill_between(steps,P_plot_test_max2,P_plot_test_min2, alpha = 0.5, color = colors[6], label='P33')
    plt.fill_between(steps,P_plot_test_max3,P_plot_test_min3, alpha = 0.5, color = colors[7], label='P12')
    plt.fill_between(steps,P_plot_test_max4,P_plot_test_min4, alpha = 0.5, color = colors[8], label='P13')
    plt.fill_between(steps,P_plot_test_max5,P_plot_test_min4, alpha = 0.5, color = 'green', label='P23')
    plt.plot(steps, P_plot2_1[0:145,0,0],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2_1[0:145,1,1],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2_1[0:145,2,2],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2_1[0:145,0,1],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2_1[0:145,0,2],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2_1[0:145,1,2],linestyle = 'dotted', color = 'black')
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Test1_Fill')
    plt.legend()
    plt.grid()
    plt.show()

def test2_data(model, random, random2):
    Data = np.genfromtxt('data/BCC_test2.txt')
    F_test2, P_test2, SE_test2, ERROR_test2 = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F_test2)
    F_test2 = np.reshape(F_test2,[num_cases,3,3])
    P_test2 = np.reshape(P_test2,[num_cases,3,3])
    
    rand = np.random.rand(100,3)
    random_test = R.from_euler('XYZ',rand).as_matrix()
    random_test2 = np.concatenate((np.reshape(np.identity(3),[1,3,3]),random_test),axis=0)
    
    F_rot2 = []
    for i in range(0,len(random_test)):
        F_rot2.append(random_test[i,:,:]@F_test2)
        
    for l in range (0, len(F_rot2)):
        F_test2 = np.concatenate((F_test2,F_rot2[l]),axis=0)
    
    training_input_test2 = tf.constant(F_test2, dtype='float32')
    prediction = model.predict(training_input_test2)[1]
    steps = np.arange(0,127)
    
    plt.figure(20, dpi=600)
    P_plot_2 = prediction[0:127,:,:]
    P_plot2_2 = P_test2[0:127,:,:]
    for j in range(1,len(random_test2)):
        n = 127*j
        P_plot_2 = np.concatenate((P_plot_2,np.linalg.inv(random_test2[j,:,:])@prediction[n:n+127,:,:]), axis = 0)
        P_plot2_2 = np.concatenate((P_plot2_2,np.linalg.inv(random_test2[j,:,:])@P_test2[n:n+127,:,:]), axis = 0)
    P_plot2_2 = P_plot2_2/1000
    for i in range(0,len(random_test2)):
        n = 127*i
        plt.plot(steps, P_plot_2[n:n+127,0,0], label='P'+str(i))#, color = colors[i])
        #plt.plot(steps, np.max(P_plot[n:n+175,0,0]), label='P'+str(i))#, color = colors[i])
        #plt.plot(steps, np.min(P_plot[n:n+175,0,0]), label='P'+str(i))#, color = colors[i])
        plt.plot(steps, P_plot2_2[0:127,0,0],linestyle = 'dotted', color = colors[2])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('test2_observer')
    #plt.legend()
    plt.grid()
    plt.show()
    
    P_plot_test = np.reshape(P_plot_2,[len(random_test2),127,3,3])
    P_plot_test_max0=[]
    P_plot_test_min0=[]
    P_plot_test_max1=[]
    P_plot_test_min1=[]
    P_plot_test_max2=[]
    P_plot_test_min2=[]
    P_plot_test_max3=[]
    P_plot_test_min3=[]
    P_plot_test_max4=[]
    P_plot_test_min4=[]
    P_plot_test_max5=[]
    P_plot_test_min5=[]
    for i in range(0,127):
        P_plot_test_max0.append(np.max(P_plot_test[:,i,0,0]))
        P_plot_test_min0.append(np.min(P_plot_test[:,i,0,0]))
        P_plot_test_max1.append(np.max(P_plot_test[:,i,1,1]))
        P_plot_test_min1.append(np.min(P_plot_test[:,i,1,1]))
        P_plot_test_max2.append(np.max(P_plot_test[:,i,2,2]))
        P_plot_test_min2.append(np.min(P_plot_test[:,i,2,2]))
        P_plot_test_max3.append(np.max(P_plot_test[:,i,0,1]))
        P_plot_test_min3.append(np.min(P_plot_test[:,i,0,1]))
        P_plot_test_max4.append(np.max(P_plot_test[:,i,0,2]))
        P_plot_test_min4.append(np.min(P_plot_test[:,i,0,2]))
        P_plot_test_max5.append(np.max(P_plot_test[:,i,1,2]))
        P_plot_test_min5.append(np.min(P_plot_test[:,i,1,2]))
    plt.figure(21, dpi=600)
    plt.fill_between(steps,P_plot_test_max0,P_plot_test_min0, alpha = 0.5, color = colors[4], label='P11')
    plt.fill_between(steps,P_plot_test_max1,P_plot_test_min1, alpha = 0.5, color = colors[5], label='P22')
    plt.fill_between(steps,P_plot_test_max2,P_plot_test_min2, alpha = 0.5, color = colors[6], label='P33')
    plt.fill_between(steps,P_plot_test_max3,P_plot_test_min3, alpha = 0.5, color = colors[7], label='P12')
    plt.fill_between(steps,P_plot_test_max4,P_plot_test_min4, alpha = 0.5, color = colors[8], label='P13')
    plt.fill_between(steps,P_plot_test_max5,P_plot_test_min4, alpha = 0.5, color = 'green', label='P23')
    plt.plot(steps, P_plot2_2[0:127,0,0],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2_2[0:127,1,1],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2_2[0:127,2,2],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2_2[0:127,0,1],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2_2[0:127,0,2],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2_2[0:127,1,2],linestyle = 'dotted', color = 'black')
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Test2_Fill')
    plt.legend()
    plt.grid()
    plt.show()
    
def test3_data(model, random, random2):
    Data = np.genfromtxt('data/BCC_test3.txt')
    F_test3, P_test3, SE_test3, ERROR_test3 = np.split(Data,[9,18,19], axis =1)
    num_cases = len(F_test3)
    F_test3 = np.reshape(F_test3,[num_cases,3,3])
    P_test3 = np.reshape(P_test3,[num_cases,3,3])
    
    rand = np.random.rand(100,3)
    random_test = R.from_euler('XYZ',rand).as_matrix()
    random_test2 = np.concatenate((np.reshape(np.identity(3),[1,3,3]),random_test),axis=0)
    
    F_rot3 = []
    for i in range(0,len(random_test)):
        F_rot3.append(random_test[i,:,:]@F_test3)
        
    for l in range (0, len(F_rot3)):
        F_test3 = np.concatenate((F_test3,F_rot3[l]),axis=0)
    
    training_input_test3 = tf.constant(F_test3, dtype='float32')
    prediction = model.predict(training_input_test3)[1]
    steps = np.arange(0,172)
    
    plt.figure(22, dpi=600)
    P_plot_3 = prediction[0:172,:,:]
    P_plot2_3 = P_test3[0:172,:,:]
    for j in range(1,len(random_test2)):
        n = 172*j
        P_plot_3 = np.concatenate((P_plot_3,np.linalg.inv(random_test2[j,:,:])@prediction[n:n+172,:,:]), axis = 0)
        P_plot2_3 = np.concatenate((P_plot2_3,np.linalg.inv(random_test2[j,:,:])@P_test3[n:n+172,:,:]), axis = 0)
    P_plot2_3 = P_plot2_3/1000
    for i in range(0,len(random_test2)):
        n = 172*i
        plt.plot(steps, P_plot_3[n:n+172,0,0], label='P'+str(i))#, color = colors[i])
        #plt.plot(steps, np.max(P_plot[n:n+175,0,0]), label='P'+str(i))#, color = colors[i])
        #plt.plot(steps, np.min(P_plot[n:n+175,0,0]), label='P'+str(i))#, color = colors[i])
        plt.plot(steps, P_plot2_3[0:172,0,0],linestyle = 'dotted', color = colors[2])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('test3_observer')
    #plt.legend()
    plt.grid()
    plt.show()
    
    P_plot_test = np.reshape(P_plot_3,[len(random_test2),172,3,3])
    P_plot_test_max0=[]
    P_plot_test_min0=[]
    P_plot_test_max1=[]
    P_plot_test_min1=[]
    P_plot_test_max2=[]
    P_plot_test_min2=[]
    P_plot_test_max3=[]
    P_plot_test_min3=[]
    P_plot_test_max4=[]
    P_plot_test_min4=[]
    P_plot_test_max5=[]
    P_plot_test_min5=[]
    for i in range(0,172):
        P_plot_test_max0.append(np.max(P_plot_test[:,i,0,0]))
        P_plot_test_min0.append(np.min(P_plot_test[:,i,0,0]))
        P_plot_test_max1.append(np.max(P_plot_test[:,i,1,1]))
        P_plot_test_min1.append(np.min(P_plot_test[:,i,1,1]))
        P_plot_test_max2.append(np.max(P_plot_test[:,i,2,2]))
        P_plot_test_min2.append(np.min(P_plot_test[:,i,2,2]))
        P_plot_test_max3.append(np.max(P_plot_test[:,i,0,1]))
        P_plot_test_min3.append(np.min(P_plot_test[:,i,0,1]))
        P_plot_test_max4.append(np.max(P_plot_test[:,i,0,2]))
        P_plot_test_min4.append(np.min(P_plot_test[:,i,0,2]))
        P_plot_test_max5.append(np.max(P_plot_test[:,i,1,2]))
        P_plot_test_min5.append(np.min(P_plot_test[:,i,1,2]))
    plt.figure(23, dpi=600)
    plt.fill_between(steps,P_plot_test_max0,P_plot_test_min0, alpha = 0.5, color = colors[4], label='P11')
    plt.fill_between(steps,P_plot_test_max1,P_plot_test_min1, alpha = 0.5, color = colors[5], label='P22')
    plt.fill_between(steps,P_plot_test_max2,P_plot_test_min2, alpha = 0.5, color = colors[6], label='P33')
    plt.fill_between(steps,P_plot_test_max3,P_plot_test_min3, alpha = 0.5, color = colors[7], label='P12')
    plt.fill_between(steps,P_plot_test_max4,P_plot_test_min4, alpha = 0.5, color = colors[8], label='P13')
    plt.fill_between(steps,P_plot_test_max5,P_plot_test_min4, alpha = 0.5, color = 'green', label='P23')
    plt.plot(steps, P_plot2_3[0:172,0,0],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2_3[0:172,1,1],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2_3[0:172,2,2],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2_3[0:172,0,1],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2_3[0:172,0,2],linestyle = 'dotted', color = 'black')
    plt.plot(steps, P_plot2_3[0:172,1,2],linestyle = 'dotted', color = 'black')
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Test3_Fill')
    plt.legend()
    plt.grid()
    plt.show()

  
  