#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 22:59:40 2021

@author: rohit
"""
import gdal
import os
import matplotlib.pyplot as plt
import numpy as np
os.chdir('/home/rohit/TDP/codes/')
from imgProcessing import hist_stretch, std_clip

data_dir = '/home/rohit/TDP/data/GeoLinked/211890821_193125911'

files = os.listdir(data_dir)
opt_file = os.path.join(data_dir,'optical_final.tif')
sar_file = os.path.join(data_dir,'sar_final.tif')
unique_tag = data_dir.split('/')[-1]
#Optical data tiling
opt_data = gdal.Open(os.path.join(data_dir,opt_file)).ReadAsArray()
# Visualization of data
# data = np.stack((opt_data[-1],opt_data[-2],opt_data[-3]))
# data  = np.einsum('ijk->jki',data)
# plt.imsave('/home/rohit/opt.png',hist_stretch(data))
shape = np.shape(opt_data)
os.chdir('/home/rohit/TDP/data/tiles/optical/')
for i  in range(shape[1]//512):
    for j in range(shape[2]//512):
       tile = opt_data[:,i*512:(i+1)*512,j*512:(j+1)*512]
       tile = np.stack((tile[-1],tile[-2],tile[-3]))
       tile = np.einsum('ijk->jki',tile)
       if not tile.max() == tile.min():
           image = hist_stretch(tile)
           plt.imsave(f'opt_{unique_tag}_{i}_{j}.jpg',image)
#SAR data tiling
sar_data = gdal.Open(os.path.join(data_dir,sar_file)).ReadAsArray()
sar_data[sar_data<0]=0
# Visualization of data
# data = hist_stretch(std_clip(sar_data))
# plt.imsave('/home/rohit/sar.png',data,cmap='gray')
sar_data = hist_stretch(std_clip(sar_data))
shape = np.shape(sar_data)
os.chdir('/home/rohit/TDP/data/tiles/sar/')
for i  in range(shape[0]//512):
    for j in range(shape[1]//512):
       tile = sar_data[i*512:(i+1)*512,j*512:(j+1)*512]
       if not tile.max() == tile.min():
           plt.imsave(f'sar_{unique_tag}_{i}_{j}.jpg',tile,cmap='gray')