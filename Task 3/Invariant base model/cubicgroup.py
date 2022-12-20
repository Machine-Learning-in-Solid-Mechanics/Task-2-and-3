# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:48:22 2022

@author: koerbel
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

def cubicgroup():
    ident = np.reshape(np.identity(3),[1,3,3])
    
    rot = np.array([np.pi/2, np.pi, 3*np.pi/2])
    rotx0 = np.reshape(R.from_euler('XYZ', np.array([rot[0],0,0])).as_matrix(),[1,3,3])
    rotx1 = np.reshape(R.from_euler('XYZ', np.array([rot[1],0,0])).as_matrix(),[1,3,3])
    rotx2 = np.reshape(R.from_euler('XYZ', np.array([rot[2],0,0])).as_matrix(),[1,3,3])
    
    roty0 = np.reshape(R.from_euler('XYZ', np.array([0,rot[0],0])).as_matrix(),[1,3,3])
    roty1 = np.reshape(R.from_euler('XYZ', np.array([0,rot[1],0])).as_matrix(),[1,3,3])
    roty2 = np.reshape(R.from_euler('XYZ', np.array([0,rot[2],0])).as_matrix(),[1,3,3])
    
    rotz0 = np.reshape(R.from_euler('XYZ', np.array([0,0,rot[0]])).as_matrix(),[1,3,3])
    rotz1 = np.reshape(R.from_euler('XYZ', np.array([0,0,rot[1]])).as_matrix(),[1,3,3])
    rotz2 = np.reshape(R.from_euler('XYZ', np.array([0,0,rot[2]])).as_matrix(),[1,3,3])
    
    rotpi0 = np.reshape(R.from_rotvec([np.pi,np.pi,0]).as_matrix(),[1,3,3])
    rotpi1 = np.reshape(R.from_rotvec([-np.pi,np.pi,0]).as_matrix(),[1,3,3])
    rotpi2 = np.reshape(R.from_rotvec([np.pi,0,np.pi]).as_matrix(),[1,3,3])
    rotpi3 = np.reshape(R.from_rotvec([-np.pi,0,np.pi]).as_matrix(),[1,3,3])
    rotpi4 = np.reshape(R.from_rotvec([0,np.pi,np.pi]).as_matrix(),[1,3,3])
    rotpi5 = np.reshape(R.from_rotvec([0,-np.pi,np.pi]).as_matrix(),[1,3,3])
    
    rot2pi30 = np.reshape(R.from_rotvec([2*np.pi/3,2*np.pi/3,2*np.pi/3]).as_matrix(),[1,3,3])
    rot2pi31 = np.reshape(R.from_rotvec([-2*np.pi/3,2*np.pi/3,2*np.pi/3]).as_matrix(),[1,3,3])
    rot2pi32 = np.reshape(R.from_rotvec([2*np.pi/3,-2*np.pi/3,2*np.pi/3]).as_matrix(),[1,3,3])
    rot2pi33 = np.reshape(R.from_rotvec([-2*np.pi/3,-2*np.pi/3,2*np.pi/3]).as_matrix(),[1,3,3])
    
    rot4pi30 = np.reshape(R.from_rotvec([4*np.pi/3,4*np.pi/3,4*np.pi/3]).as_matrix(),[1,3,3])
    rot4pi31 = np.reshape(R.from_rotvec([-4*np.pi/3,4*np.pi/3,4*np.pi/3]).as_matrix(),[1,3,3])
    rot4pi32 = np.reshape(R.from_rotvec([4*np.pi/3,-4*np.pi/3,4*np.pi/3]).as_matrix(),[1,3,3])
    rot4pi33 = np.reshape(R.from_rotvec([-4*np.pi/3,-4*np.pi/3,4*np.pi/3]).as_matrix(),[1,3,3])
    
    G7 = np.concatenate((ident,rotx0,rotx1,rotx2,roty0,roty1,roty2,rotz0,rotz1,rotz2,rotpi0,rotpi1,rotpi2,rotpi3,rotpi4,rotpi5,rot2pi30,rot2pi31,rot2pi32,rot2pi33,rot4pi30,rot4pi31,rot4pi32,rot4pi33), axis=0)
    
    rand = np.random.rand(8,3)
    random = R.from_euler('XYZ',rand).as_matrix()
    random2 = np.concatenate((np.reshape(np.identity(3),[1,3,3]),random),axis=0)
    

    return G7,random, random2