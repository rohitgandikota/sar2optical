#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:30:58 2021

@author: Rohit Gandikota
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
os.chdir('/home/rohit/TDP/codes')
from ResAutoModel import RemoteSenseNet,RemoteSenseNetV2,RemoteSenseUNet
from CustomDataAugmentation import augmentation
import tensorflow as tf
from imgProcessing import hist_stretch_set
import datetime
from VGG19 import VGG19
#%% Data Reading and Cleaning
data_path = '/appdisk/TDP/data/tiles'
os.chdir(data_path)
files = os.listdir(os.path.join(data_path,'optical'))
opt_data = []
sar_data = []
for file in files:
    try:
        opt = plt.imread(data_path+'/optical/'+file)
        sar = np.expand_dims(plt.imread(data_path+'/sar/sar'+file[3:])[:,:,0],axis=-1)      
    except Exception as e:
        print(e)
        continue
    opt_data.append(opt)
    sar_data.append(sar)
opt_data = np.array(opt_data) # (number tiles,x sizw,y size, channels size)
sar_data = np.array(sar_data)# (number tiles,x sizw,y size, channels size)


X, Y = augmentation(opt_data,sar_data)

# data  = np.concatenate((opt_data,sar_data),axis=-1)
# datagen = ImageDataGenerator(rotation_range=90)
# # prepare iterator
# it = datagen.flow(data, batch_size=1)

# new_data = []
# # generate samples and plot
# for i in range(5000):
# 	# generate batch of images
# 	batch = it.next()
# 	# convert to unsigned integers for viewing
# 	image = batch[0].astype('uint8')
# 	# plot raw pixel data
# 	new_data.append(image)
# a = np.vstack([data,new_data])
# new_data = np.array(new_data)
# new_data = np.vstack([data,new_data])
# X = new_data[:,:,:,:3]
# Y = new_data[:,:,:,-1:]
    
    
del(opt_data,sar_data)
X_train = X[:2000,:,:,:]
X_train = hist_stretch_set(X_train,scale=255)
Y_train = Y[:2000,:,:,:]
Y_train = hist_stretch_set(Y_train,scale=255)
X_test = X[2000:,:,:,:]
X_test = hist_stretch_set(X_test,scale=255)
Y_test = Y[2000:,:,:,:]
Y_test = hist_stretch_set(Y_test,scale=255)
# cleaning for nan
X_train_new = []
Y_train_new = []
for i in range(len(X_train)):
    if  X_train[i][np.isnan(X_train[i])] != [] or Y_train[i][np.isnan(Y_train[i])] != []:
        continue
    X_train_new.append(X_train[i])
    Y_train_new.append(Y_train[i])
# cleaning for nan
X_test_new = []
Y_test_new = []
for i in range(len(X_test)):
    if  X_test[i][np.isnan(X_test[i])] != [] or Y_test[i][np.isnan(Y_test[i])] != []:
        continue
    X_test_new.append(X_test[i])
    Y_test_new.append(Y_test[i])
    
del(X_train, Y_train, X_test, Y_test, X, Y)

X_test = np.array(X_test_new)
Y_test = np.array(Y_test_new)
X_train = np.array(X_train_new)
Y_train = np.array(Y_train_new)
del(X_test_new,Y_test_new,X_train_new, Y_train_new )
#%% Model Initialisation and set-up
model = RemoteSenseNet()
optimizer = Adam(0.0001,0.5)
optimizer1 = Adam(0.001, 0.5)
# Metric functtion
def ssim_loss(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
model.compile(loss=['mse'], optimizer=optimizer,metrics=[ssim_loss])
# checkpoint
# model.summary()
# filepath="/appdisk/TDP/models/deepNetTrainingpy/weights-improvement-{epoch:02d}-{val_ssim_loss:.2f}.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_ssim_loss', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]
# # Fit the model
# history = model.fit(Y_train, X_train, validation_split=0.3, epochs=1500, batch_size=10, callbacks=callbacks_list, verbose=True)

# Plot train results
def saveImgs(x, y, model, fname=''):
    fig = plt.figure(figsize=(15,15))
    img = model.predict(x)
    for i in range(len(x)):
        plt.subplot(3,3,(i*3)+1)
        
        # plt.imshow(cv2.cvtColor(g_img[0], cv2.COLOR_RGB2BGR))
        plt.imshow(np.uint8(img[i]*255))
        plt.axis('off')
        plt.title('Generated Image')
        
        plt.subplot(3,3,(i*3)+2)
        # plt.imshow(cv2.cvtColor(c_img[0], cv2.COLOR_RGB2BGR))
        plt.imshow(np.uint8(y[i]*255),cmap='gray')
        plt.axis('off')
        plt.title('Optical Image')
        
        plt.subplot(3,3,(i*3)+3)
        # plt.imshow(cv2.cvtColor(c_img[0], cv2.COLOR_RGB2BGR))
        plt.imshow(np.uint8(x[i].reshape(512,512)*255),cmap='gray')
        plt.axis('off')
        plt.title('SAR Image')
    fig.savefig(fname)
# Batch training
Loss = []

select = []
epochs = 100000
for epoch in range(epochs):
    start_time = datetime.datetime.now()
    x = X_train[epoch%80*50:(epoch%80+1)*50]
    y = Y_train[epoch%80*50:(epoch%80+1)*50]
    # start_time = datetime.datetime.now()
    auto_loss = model.train_on_batch(y,x)
    if epoch == 0:
        min_loss = auto_loss[0]
        max_ssim = auto_loss[1]
    Loss.append(auto_loss)
    elapsed_time = datetime.datetime.now() - start_time
    print ("[Epoch %d/%d]  [Model loss: %f Model SSIM: %f time: %s" % (epoch, epochs, auto_loss[0], auto_loss[1], elapsed_time))
    if min_loss > auto_loss[0]:
        answer = 'WithOut'
        min_loss = auto_loss[0]
        if max_ssim < auto_loss[1]:
            answer = 'With'     
            max_ssim = auto_loss[1]
        select.append([epoch, auto_loss[0],auto_loss[1]]) 
        print(f'Improvement found in MSE {answer} SSIM Improvement !!!')
        model.save(f'/appdisk/TDP/models/deepNetTrainingpy/Model_epoch{epoch}_mseLoss_{str(round(min_loss,3)).replace(".","_")}_{answer}SSIMImprovement.h5')
        saveImgs(y[:3],x[:3],model,fname=f'/appdisk/TDP/testBed/deepNetTrainingpy/ImprovedEpoch_{epoch}_{answer}SSIMImprovement.png')
    if (epoch%500 ==0) :
        # Check on extracter loss
        select.append([epoch, auto_loss[0],auto_loss[1]])
        # serialize weights to HDF5
        model.save(f'/appdisk/TDP/models/deepNetTrainingpy/Model_epoch{epoch}_mseLoss_{str(round(min_loss,3)).replace(".","_")}.h5')
        print("Saved model to disk")
        saveImgs(y[:3],x[:3],model,fname=f'/appdisk/TDP/testBed/deepNetTrainingpy/Epoch_{epoch}.png')
 
fig = plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.plot(np.array(Loss)[:,0])
plt.title('Training MSE')
plt.subplot(2,2,2)
plt.plot(np.array(Loss)[:,1])
plt.title('Training SSIM')
plt.subplot(2,2,3)
plt.plot(np.array(select)[:,1])
plt.title('Validation MSE')
plt.subplot(2,2,4)
plt.plot(np.array(select)[:,2])
plt.title('Validation SSIM')
fig.savefig('/appdisk/TDP/testBed/deepNetTrainingpy/firstTraining10kepochs.png')       

#%% Model V2 with feature Loss from VGG19. Initialisation and set-up
model, disc = RemoteSenseNetV2()  # two outputs [actual output, feature output from VGG19 intermediate layer]

optimizer = Adam(0.0001,0.5)
optimizer1 = Adam(0.001, 0.5)
# Metric functtion
def ssim_loss(y_true, y_pred):
  return -1*tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0)) + tf.keras.losses.MeanAbsoluteError()(y_true,y_pred)

model.compile(loss=['binary_crossentropy', ssim_loss,'mse'], loss_weights=[1,1,5], optimizer=optimizer)
disc.compile(loss='mse', optimizer=optimizer, metrics=['accuracy']) 


# Plot train results
def saveImgs(x, y, model, fname=''):
    fig = plt.figure(figsize=(10,10))
    img = model.predict(x)[1]
    for i in range(len(x)):
        plt.subplot(1,3,(i*3)+1)
        
        # plt.imshow(cv2.cvtColor(g_img[0], cv2.COLOR_RGB2BGR))
        plt.imshow(np.uint8(img[i]*255))
        plt.axis('off')
        plt.title('Generated Image')
        
        plt.subplot(1,3,(i*3)+2)
        # plt.imshow(cv2.cvtColor(c_img[0], cv2.COLOR_RGB2BGR))
        plt.imshow(np.uint8(y[i]*255),cmap='gray')
        plt.axis('off')
        plt.title('Optical Image')
        
        plt.subplot(1,3,(i*3)+3)
        # plt.imshow(cv2.cvtColor(c_img[0], cv2.COLOR_RGB2BGR))
        plt.imshow(np.uint8(x[i].reshape(512,512)*255),cmap='gray')
        plt.axis('off')
        plt.title('SAR Image')
    fig.savefig(fname)


## Batch training
Loss = []
vgg19 = VGG19(include_top=False, weights='imagenet')
select = []
epochs = 20000000
for epoch in range(epochs):
    start_time = datetime.datetime.now()
    x = X_train[epoch%1974:epoch%1974+1]
    y = Y_train[epoch%1974:epoch%1974+1]
    fakes = np.zeros((len(y),1))
    trues = np.ones((len(y),1))
    features = vgg19.predict(x)
    # start_time = datetime.datetime.now()
    
    
    auto_loss = model.train_on_batch(y,[trues, x,features])
    
    disc.trainable=True
    disc_loss = disc.train_on_batch(x,trues)
    disc_loss = disc.train_on_batch(x,trues)

    generated = model.predict(y)[1]
    disc_loss = disc.train_on_batch(generated,fakes)
    disc_loss = disc.train_on_batch(generated,fakes)
    disc.trainable=False
    
    if epoch == 0:
        min_loss = auto_loss[0]
        min_feature = auto_loss[2]
    Loss.append(auto_loss)
    elapsed_time = datetime.datetime.now() - start_time
    print ("[Epoch %d/%d]  [Model loss: %f Generator Loss: %f Discriminator Loss: %f SSIM+MAE: %f Feature loss: %f time: %s" % (epoch, epochs, auto_loss[0], auto_loss[1], disc_loss[0], auto_loss[2], auto_loss[3], elapsed_time))
    if min_loss > auto_loss[0]:
        answer = 'WithOut'
        min_loss = auto_loss[0]
        if min_feature > auto_loss[3]:
            answer = 'With'     
            min_feature = auto_loss[3]
        select.append([epoch,auto_loss[0], auto_loss[2],auto_loss[3]]) 
        print(f'Improvement found in MSE {answer} Deep Feature Improvement !!!')
        model.save(f'/appdisk/TDP/models/deepNetTrainingpyV2/Model_epoch{epoch}_mseLoss_{str(round(min_loss,3)).replace(".","_")}_{answer}DeepFeaturesImprovement.h5')
        saveImgs(y[:3],x[:3],model,fname=f'/appdisk/TDP/testBed/deepNetTrainingpyV2/ImprovedLoss_{epoch}_{answer}DeepFeaturesImprovement.png')
    if (epoch%500 ==0) :
        # Check on extracter loss
        select.append([epoch,auto_loss[0], auto_loss[2],auto_loss[3]])

        # serialize weights to HDF5
        model.save(f'/appdisk/TDP/models/deepNetTrainingpyV2/Model_epoch{epoch}_mseLoss_{str(round(min_loss,3)).replace(".","_")}.h5')
        print("Saved model to disk")
        saveImgs(y,x,model,fname=f'/appdisk/TDP/testBed/deepNetTrainingpyV2/Epoch_{epoch}.png')
 
fig = plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
plt.plot(np.array(Loss)[:,0])
plt.title('Training Total Loss')
plt.subplot(2,3,2)
plt.plot(np.array(Loss)[:,1])
plt.title('Training SSIM+MAE Loss')
plt.subplot(2,3,3)
plt.plot(np.array(Loss)[:,2])
plt.title('Training Deep Feature Loss')

plt.subplot(2,3,4)
plt.plot(np.array(select)[:,0])
plt.title('Validation Total Loss')
plt.subplot(2,3,5)
plt.plot(np.array(select)[:,1])
plt.title('Validation SSIM+MAE Loss')
plt.subplot(2,3,6)
plt.plot(np.array(select)[:,2])
plt.title('Validation Deep Feature Loss')

fig.savefig('/appdisk/TDP/testBed/deepNetTrainingpyV2/aV2Training100kepochs.png')


#%% Model UNet  Initialisation and set-up
model, disc = RemoteSenseNetV2(outBands=1)  # two outputs [actual output, feature output from VGG19 intermediate layer]

optimizer = Adam(0.0001,0.5)
optimizer1 = Adam(0.001, 0.5)
# Metric functtion
def ssim_loss(y_true, y_pred):
  return -1*tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0)) + tf.keras.losses.MeanAbsoluteError()(y_true,y_pred)

model.compile(loss=['binary_crossentropy', ssim_loss,'mse'], loss_weights=[1,1,5], optimizer=optimizer)
disc.compile(loss='mse', optimizer=optimizer, metrics=['accuracy']) 

def rgb2gray(rgb):

    r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return np.expand_dims(gray,axis=-1)
# Plot train results
def saveImgsGray(x, y, model, fname=''):
    fig = plt.figure(figsize=(10,10))
    img = model.predict(x)[1]
    for i in range(len(x)):
        plt.subplot(1,3,(i*3)+1)
        
        # plt.imshow(cv2.cvtColor(g_img[0], cv2.COLOR_RGB2BGR))
        plt.imshow(np.uint8(img[i,:,:,0]*255),cmap='gray')
        plt.axis('off')
        plt.title('Generated Image')
        
        plt.subplot(1,3,(i*3)+2)
        # plt.imshow(cv2.cvtColor(c_img[0], cv2.COLOR_RGB2BGR))
        plt.imshow(np.uint8(y[i]*255),cmap='gray')
        plt.axis('off')
        plt.title('Optical Image')
        
        plt.subplot(1,3,(i*3)+3)
        # plt.imshow(cv2.cvtColor(c_img[0], cv2.COLOR_RGB2BGR))
        plt.imshow(np.uint8(x[i].reshape(512,512)*255),cmap='gray')
        plt.axis('off')
        plt.title('SAR Image')
    fig.savefig(fname)

## Batch training
Loss = []
select = []
vgg19 = VGG19(include_top=False, weights='imagenet')
epochs = 20000000
for epoch in range(epochs):
    start_time = datetime.datetime.now()
    x = X_train[epoch%1974:epoch%1974+1]
    y = Y_train[epoch%1974:epoch%1974+1]
    features = vgg19.predict(x)
    x = rgb2gray(x)
    fakes = np.zeros((len(y),1))
    trues = np.ones((len(y),1))
    
    # start_time = datetime.datetime.now()
    
    
    auto_loss = model.train_on_batch(y,[trues, x,features])
    
    disc.trainable=True
    disc_loss = disc.train_on_batch(x,trues)
    disc_loss = disc.train_on_batch(x,trues)

    generated = model.predict(y)[1]
    disc_loss = disc.train_on_batch(generated,fakes)
    disc_loss = disc.train_on_batch(generated,fakes)
    disc.trainable=False    
   
       
    if epoch == 0:
        min_loss = auto_loss[0]
        min_feature = auto_loss[2]
    Loss.append(auto_loss)
    elapsed_time = datetime.datetime.now() - start_time
    print ("[Epoch %d/%d]  [Model loss: %f Generator Loss: %f Discriminator Loss: %f SSIM+MAE: %f Feature loss: %f time: %s" % (epoch, epochs, auto_loss[0], auto_loss[1], disc_loss[0], auto_loss[2], auto_loss[3], elapsed_time))
    if min_loss > auto_loss[0]:
        answer = 'WithOut'
        min_loss = auto_loss[0]
        if min_feature > auto_loss[3]:
            answer = 'With'     
            min_feature = auto_loss[3]
        select.append([epoch,auto_loss[0], auto_loss[2],auto_loss[3]]) 
        print(f'Improvement found in MSE {answer} Deep Feature Improvement !!!')
        model.save(f'/appdisk/TDP/models/deepNetTrainingpyV2Gray/Model_epoch{epoch}_Gray_GAN_mseLoss_{str(round(min_loss,3)).replace(".","_")}_{answer}DeepFeaturesImprovement.h5')
        saveImgsGray(y,x,model,fname=f'/appdisk/TDP/testBed/deepNetTrainingpyV2Gray/ImprovedLoss_{epoch}_{answer}DeepFeaturesImprovement_Gray_GAN.png')
    if (epoch%500 ==0) :
        # Check on extracter loss
        select.append([epoch,auto_loss[0], auto_loss[2],auto_loss[3]])

        # serialize weights to HDF5
        if epoch%50000 == 0:
            model.save(f'/appdisk/TDP/models/deepNetTrainingpyV2Gray/Model_epoch{epoch}_Gray_GAN_mseLoss_{str(round(min_loss,3)).replace(".","_")}.h5')
        print("Saved model to disk")
        saveImgsGray(y,x,model,fname=f'/appdisk/TDP/testBed/deepNetTrainingpyV2Gray/Epoch_{epoch}_Gray_GAN.png')
        
fig = plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
plt.plot(np.array(Loss)[:,0])
plt.title('Training Total Loss')
plt.subplot(2,3,2)
plt.plot(np.array(Loss)[:,1])
plt.title('Training SSIM+MAE Loss')
plt.subplot(2,3,3)
plt.plot(np.array(Loss)[:,2])
plt.title('Training Deep Feature Loss')

plt.subplot(2,3,4)
plt.plot(np.array(select)[:,0])
plt.title('Validation Total Loss')
plt.subplot(2,3,5)
plt.plot(np.array(select)[:,1])
plt.title('Validation SSIM+MAE Loss')
plt.subplot(2,3,6)
plt.plot(np.array(select)[:,2])
plt.title('Validation Deep Feature Loss')

fig.savefig('/appdisk/TDP/testBed/deepNetTrainingpyV2Gray/aGrayGANTraining4Croreepochs.png')