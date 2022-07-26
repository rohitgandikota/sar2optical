#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 10:06:05 2021

@author: Rohit Gandikota
"""
import numpy as np
from skimage.transform import rescale
def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx, :]
def augmentation(X,Y):
    new_X = []
    new_Y = []
    
    for i in range(len(X)):
        rot90 = np.rot90(X[i], k=1, axes=(0, 1))
        rot270 = np.rot90(X[i], k=-1, axes=(0, 1))
        flipL = np.fliplr(X[i])
        flipU = np.flipud(X[i])
        scale_out = rescale(X[i], scale=2.0, mode='constant',multichannel=(True))
        scale_out = crop_center(scale_out,np.shape(X[i])[0],np.shape(X[i])[1])
        new_X.extend([rot90,rot270,X[i],flipL,flipU,scale_out])
        
        rot90 = np.rot90(Y[i], k=1, axes=(0, 1))
        rot270 = np.rot90(Y[i], k=-1, axes=(0, 1))
        flipL = np.fliplr(Y[i])
        flipU = np.flipud(Y[i])
        scale_out = rescale(Y[i], scale=2.0, mode='constant',multichannel=(True))
        scale_out = crop_center(scale_out,np.shape(Y[i])[0],np.shape(Y[i])[1])
        new_Y.extend([rot90,rot270,Y[i],flipL,flipU,scale_out])
        
    return np.array(new_X) , np.array(new_Y)