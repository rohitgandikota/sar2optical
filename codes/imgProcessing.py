#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 23:05:51 2021

@author: rohit
"""
import numpy as np

def hist_stretch(data,scale=255):
    if scale==1:
        return ((data-data.min())/(data.max() - data.min())*scale)
    if scale == -1:
        return (2*((data-data.min())/(data.max() - data.min()))-1)
    return np.uint8((data-data.min())/(data.max() - data.min())*scale)
def std_clip(data):
    mean = np.mean(data)
    std = np.std(data)
    return np.clip(data,max(0,mean-(std*2)),mean+(std*2))

def hist_stretch_set(data,scale=255):
    data = np.float16(data)
    new = []
    for i in range(len(data)):
        if scale == -1:
            new.append(hist_stretch(data[i],scale=-1))
        else:
            new.append(hist_stretch(data[i],scale=1)*scale)
    return (np.array(new))

def gaussian_stretch(data):
    
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    new = []
    for i in range(len(data)):
        new.append(np.divide((data[i]-mean),std))        
    return np.array(new)