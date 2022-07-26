#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 16:12:55 2021

@author: Rohit Gandikota
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
os.chdir('/home/rohit/TDP/codes')
from ResAutoModel import DisMonNet
from CustomDataAugmentation import augmentation
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from imgProcessing import hist_stretch_set,gaussian_stretch
import datetime
import glob
from VGG19 import VGG19
#%% Data Reading and Cleaning
data_path = '/appdisk/TDP/data/tiles'
os.chdir(data_path)
files = os.listdir(os.path.join(data_path,'optical'))
opt_data = []
sar_data = []

# Reading images
for file in files:
    try:
        opt = plt.imread(data_path+'/optical/'+file)
        sar = np.expand_dims(plt.imread(data_path+'/sar/sar'+file[3:])[:,:,0],axis=-1)      
    except Exception as e:
        print(e)
        continue
    opt_data.append(opt)
    sar_data.append(sar)
opt_data = np.array(opt_data)
sar_data = np.array(sar_data)

# data augmentation
Y, X = augmentation(opt_data,sar_data)

# Train-Test split
X_train = X[:2000,:,:,:]
X_train = gaussian_stretch(X_train)
Y_train = Y[:2000,:,:,:]
Y_train = hist_stretch_set(Y_train,scale=-1)
X_test = X[2000:,:,:,:]
X_test = gaussian_stretch(X_test)
Y_test = Y[2000:,:,:,:]
Y_test = hist_stretch_set(Y_test,scale=-1)

# Remove NaN samples
X_train_new = []
Y_train_new = []
for i in range(len(X_train)):
    if  X_train[i][np.isnan(X_train[i])] != [] or Y_train[i][np.isnan(Y_train[i])] != []:
        continue
    X_train_new.append(X_train[i])
    Y_train_new.append(Y_train[i])

X_test_new = []
Y_test_new = []
for i in range(len(X_test)):
    if  X_test[i][np.isnan(X_test[i])] != [] or Y_test[i][np.isnan(Y_test[i])] != []:
        continue
    X_test_new.append(X_test[i])
    Y_test_new.append(Y_test[i])
    
del(X_train, Y_train, X_test, Y_test, X, Y, opt_data,sar_data)

# Renaming and curating the final data
X_test = np.array(X_test_new)
Y_test = np.array(Y_test_new)
X_train = np.array(X_train_new)
Y_train = np.array(Y_train_new)

del(X_test_new,Y_test_new,X_train_new, Y_train_new )

#%% Model Initialization and training
model = DisMonNet(512,512)
optimizer1 = Adam(0.01, 0.5)
# Metric functtion
def ssim_loss(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
model.compile(loss=['mse'], optimizer=optimizer1,metrics=[ssim_loss])
# checkpoint
# model.summary()
filepath="/appdisk/TDP/models/DisMonNet/weights-improvement-{epoch:02d}-{val_ssim_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_ssim_loss', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# Fit the model
history = model.fit(X_train, Y_train, validation_split=0.3, epochs=100000, batch_size=10, callbacks=callbacks_list, verbose=True)

# Visualise training curve
fig = plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.plot(np.array(history.history['train_loss']))
plt.title('Training MSE')
plt.subplot(2,2,2)
plt.plot(history.history['train_acc'])
plt.title('Training SSIM')
plt.subplot(2,2,3)
plt.plot(history.history['val_loss'])
plt.title('Validation MSE')
plt.subplot(2,2,4)
plt.plot(history.history['val_acc'])
plt.title('Validation SSIM')
fig.savefig('/appdisk/TDP/testBed/DisMonNet/firstTraining10kepochs.png')       


# Display train results
def saveImgs(x, y, model, fname=''):
    fig = plt.figure(figsize=(15,15))
    img = model.predict(x)
    for i in range(len(x)):
        plt.subplot(3,3,(i*3)+1)
        
        # plt.imshow(cv2.cvtColor(g_img[0], cv2.COLOR_RGB2BGR))
        plt.imshow(np.uint8(((img[i]+1)/2)*255))
        plt.axis('off')
        plt.title('Generated Image')
        
        plt.subplot(3,3,(i*3)+2)
        # plt.imshow(cv2.cvtColor(c_img[0], cv2.COLOR_RGB2BGR))
        plt.imshow(np.uint8(((y[i]+1)/2)*255))
        plt.axis('off')
        plt.title('Optical Image')
        
        plt.subplot(3,3,(i*3)+3)
        # plt.imshow(cv2.cvtColor(c_img[0], cv2.COLOR_RGB2BGR))
        plt.imshow(np.uint8(x[i].reshape(512,512)*255),cmap='gray')
        plt.axis('off')
        plt.title('SAR Image')
    fig.savefig(fname)




model_weights = glob.glob('/appdisk/TDP/models/DisMonNet/*.h5')
for weight in model_weights:
    model.load_weights(weight)
    dir_name = weight.split('/')[-1].split('.')[0]
    os.mkdir(f'/appdisk/TDP/testBed/DisMonNet/{dir_name}')
    for index in np.arange(0,len(X_test),3):    
        try:
            saveImgs(X_test[index:index+3], Y_test[index:index+3], model, fname=f'/appdisk/TDP/testBed/DisMonNet/{dir_name}/Test_{index}')
        except:
            pass
    



