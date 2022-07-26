# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:54:22 2020

@author: Rohit Gandikota
"""
import glob
import gdal
import numpy as np
import os
os.chdir('/home/rohit/TDP/codes')
from ResAutoModel import RemoteSenseNetV2
from imgProcessing import hist_stretch,std_clip


## Tiff writing with geo-referencing
def writeGeoTiff(InputArray, NBANDS, NROWS, NCOLS, wkt_projection,geotransform,fname):
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(fname, NCOLS, NROWS, NBANDS, gdal.GDT_UInt16)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(wkt_projection)
    for i in range(NBANDS):
        dataset.GetRasterBand(i+1).WriteArray(InputArray[:,:,i])
    dataset.FlushCache()
    return None

test_save_path = '/appdisk/TDP/testBed/deepNetTrainingpyV2Gray/'
sar_path = '/home/rohit/TDP/data/Data_urban/SAR_Data_Urban/193126521_VV.tif'
sar_file_name = sar_path.split('/')[-1]
#SAR data tiling
sar_dataset = gdal.Open(sar_path)
sar_data = sar_dataset.ReadAsArray()
sar_data[sar_data<0]=0
sar_data = hist_stretch(std_clip(sar_data),scale=1)
shape = np.shape(sar_data)
# Test models loading 
test_epochs = ['375000']

model, disc = RemoteSenseNetV2(outBands=1)  
del(disc)
weights_list = []
for epoch in test_epochs:
    weights = glob.glob(f'/appdisk/TDP/models/deepNetTrainingGANpyV2/Model_epoch{epoch}*')
    print('\n**********************************')
    print(f'Loading the model : {weights[0]}')
    weights_list.extend(weights)
    model.load_weights(weights[0])
    opt_data = np.zeros((shape[0],shape[1],1))
    for i  in range(shape[0]//512):
        for j in range(shape[1]//512):
           tile = sar_data[i*512:(i+1)*512,j*512:(j+1)*512]
           opt = model.predict(np.expand_dims(tile,axis=0))
           opt_data[i*512:(i+1)*512,j*512:(j+1)*512] = opt[0]
    opt_data[sar_data==0]=0
    print('Prediction of the SAR to Optical is complete, saving the files for reference')
    print('**********************************\n ')
    writeGeoTiff(opt_data*255, 1, shape[0], shape[1],sar_dataset.GetProjection(),sar_dataset.GetGeoTransform(), fname=f'{test_save_path}/Predict_{epoch}_{sar_file_name}')
    
    