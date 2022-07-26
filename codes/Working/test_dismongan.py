#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:53:27 2021

@author: Rohit Gandikota
"""
# example of loading a pix2pix model and using it for image to image translation
from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
import numpy as np
import os
import glob
 
# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	titles = ['Source', 'Generated', 'Expected']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 3, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# show title
		pyplot.title(titles[i])
	pyplot.show()

# Display train results
def saveImgs(x, y, model, fname=''):
    fig = pyplot.figure(figsize=(15,15))
    img = model.predict(x)
    for i in range(len(x)):
        pyplot.subplot(3,3,(i*3)+1)
        
        # pyplot.imshow(cv2.cvtColor(g_img[0], cv2.COLOR_RGB2BGR))
        pyplot.imshow(np.uint8(((img[i]+1)/2)*255))
        pyplot.axis('off')
        pyplot.title('Generated Image')
        
        pyplot.subplot(3,3,(i*3)+2)
        # pyplot.imshow(cv2.cvtColor(c_img[0], cv2.COLOR_RGB2BGR))
        pyplot.imshow(np.uint8(((y[i]+1)/2)*255))
        pyplot.axis('off')
        pyplot.title('Optical Image')
        
        pyplot.subplot(3,3,(i*3)+3)
        # pyplot.imshow(cv2.cvtColor(c_img[0], cv2.COLOR_RGB2BGR))
        pyplot.imshow(np.uint8(x[i].reshape(512,512)*255),cmap='gray')
        pyplot.axis('off')
        pyplot.title('SAR Image')
    fig.savefig(fname)

# load dataset
os.chdir('/appdisk/TDP/data/SpyderVariables')
X_test = load('X_test.npy')
Y_test = load('Y_test.npy')
print('Loaded', X_test.shape, Y_test.shape)
# load model
model_weights = glob.glob('/appdisk/TDP/models/DisMonGAN/*.h5')
for weight in model_weights:
    model = load_model(weight)
    dir_name = weight.split('/')[-1].split('.')[0]
    os.mkdir(f'/appdisk/TDP/testBed/DisMonGAN/Test/{dir_name}')
    for index in np.arange(0,len(X_test),3):    
        try:
            saveImgs(X_test[index:index+3], Y_test[index:index+3], model, fname=f'/appdisk/TDP/testBed/DisMonGAN/Test/{dir_name}/Test_{index}')
        except:
            pass
# model = load_model('model_109600.h5')
# # select random example
# ix = randint(0, len(X_test), 1)
# src_image, tar_image = X_test[ix], Y_test[ix]
# # generate image from source
# gen_image = model.predict(src_image)
# # plot all three images
# plot_images(src_image, gen_image, tar_image)