#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:28:29 2021

@author: Rohit Gandikota
"""
from keras.layers import *
import keras
from keras.models import Model, Sequential
from VGG19 import VGG19
from keras.optimizers import Adam
def ResEncoder(inp, nfilter, ksize=3, stride=2, bn = True):
    
    x = Conv2D(int(nfilter/2), ksize, strides = 1, padding= 'same')(inp)
    x = LeakyReLU()(x)    
    x = BatchNormalization()(x)
    
    x = Conv2D(nfilter, ksize, strides = stride, padding= 'same')(x)   
    x = LeakyReLU()(x)  
    x = BatchNormalization()(x)
    
    x1 = Conv2D(nfilter, ksize, strides = stride, padding='same')(inp)
    x1 = LeakyReLU()(x1)  
    x1 = BatchNormalization()(x1)
    
    out = Add()([x,x1])
    
    return out

def ResDecoder(inp, enc_layer, nfilter, ksize=3, stride=2, bn = True):
    
    x = UpSampling2D(size=(2, 2))(inp)
    x = Conv2DTranspose(int(nfilter/2), ksize, strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(nfilter, ksize, strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x1 = UpSampling2D(size=(2, 2))(inp)
    x1 = Conv2DTranspose(nfilter, ksize, strides=(1, 1), padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    
    out = Add()([x,x1])
    skip_out = Concatenate()([out,enc_layer])
    
    return skip_out

def RemoteSenseDisc(inpBands=3):
    nf = 8
    inp_layer1 = Input(shape=(512,512,inpBands))
    # Encoder 
    x  = ResEncoder(inp_layer1, nf) # 64
    x1 = ResEncoder(x, nf*2) # 32 
    x2 = ResEncoder(x1, nf*4) # 16
    x3 = ResEncoder(x2, nf*8) # 8
    x4 = ResEncoder(x3, nf*16) # 4
    x5 = ResEncoder(x4, nf*32) # 2 
    x5 = ResEncoder(x5, nf*32)
    x5 = ResEncoder(x5, nf*64)
    x5 = Flatten()(x5)
    x6 = Dense(512,activation='relu')(x5)
    x6 = Dense(128,activation='relu')(x6)
    x6 = Dense(32,activation='relu')(x6)
    x6 = Dense(1,activation='sigmoid')(x6)
    
    model = Model(inp_layer1,x6)
    return model
def RemoteSenseNet(inpBands=1,outBands=3):
    nf = 8
    inp_layer1 = Input(shape=(None,None,inpBands))
    # Encoder 
    x  = ResEncoder(inp_layer1, nf) # 64
    x1 = ResEncoder(x, nf*2) # 32 
    x2 = ResEncoder(x1, nf*4) # 16
    x3 = ResEncoder(x2, nf*8) # 8
    x4 = ResEncoder(x3, nf*16) # 4
    x5 = ResEncoder(x4, nf*32) # 2 
    # Decoder
    x6 = ResDecoder(x5, x4, nf*16)
    x7 = ResDecoder(x6, x3, nf*8)
    x8 = ResDecoder(x7, x2, nf*4)
    x9 = ResDecoder(x8, x1, nf*2)
    x10 = ResDecoder(x9, x, nf)
    
    x11 = UpSampling2D(size=(2, 2))(x10)
    out = Conv2DTranspose(outBands, 3 , strides=(1, 1), padding='same', activation='sigmoid')(x11)
    model    = Model(inp_layer1, out)
    return model

def RemoteSenseNetV2(inpBands=1,outBands=3):
    import keras
    model = RemoteSenseNet(inpBands,outBands)
    
    disc = RemoteSenseDisc(outBands)
    optimizer = Adam(0.0001,0.5)
    disc.compile(loss='mse', optimizer=optimizer, metrics=['accuracy']) 
    
    inpLayer = Input(shape=(None,None,inpBands))
    # By conditioning on B generate a fake version of A
    outLayer = model(inpLayer)
    
    #Generating spectrogram
    
    # For the combined model we will only train the generator
    disc.trainable = False
    #    disc_feat.trainable = False
    # Discriminators determines validity of translated images / condition pairs
    valido = disc(outLayer)
    #    valido_feat = disc_feat(spect)
    vgg19 = VGG19(include_top=False, weights='imagenet')
    vgg19.trainable = False
    if outBands==1:
        band = keras.layers.concatenate([outLayer,outLayer,outLayer], axis=-1)
        featureLayer = vgg19(band)
    
    sar2opt = Model(inputs=inpLayer, outputs=[valido, outLayer,featureLayer])
    
    return sar2opt, disc

def RemoteSenseNetV2_NonGAN(inpBands=1,outBands=3):
    model = RemoteSenseNet()
    
    
    inpLayer = Input(shape=(None,None,inpBands))
    # By conditioning on B generate a fake version of A
    outLayer = model(inpLayer)
    
    vgg19 = VGG19(include_top=False, weights='imagenet')
    vgg19.trainable = False
    
    featureLayer = vgg19(outLayer)
    
    sar2opt = Model(inputs=inpLayer, outputs=[outLayer,featureLayer])
    
    return sar2opt
def RemoteSenseUNet(inpBands=1,outBands=3):
    inputs = Input((None, None, inpBands))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(outBands, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def DisMonNet(inpshapex,inpshapey, inpBands=1,outBands=3):
    def convBlock(inputs,filters=32,size=(3,3)):
        conv1 = Conv2D(filters, size, padding='same')(inputs)
        conv1 = Conv2D(filters, size, strides=(2,2), padding='same')(conv1)
        if filters>127:
            conv1 = Conv2D(filters//2, (1,1), padding='same')(conv1)
        BN1 = BatchNormalization()(conv1)
        AC1 = LeakyReLU()(BN1)
        return AC1
    
    def deconvBlock(inputs, connect, filters=32,size=(3,3)):
        x = UpSampling2D(size=(2, 2))(inputs)
        
        x = Conv2DTranspose(filters, size, strides=(1, 1), padding='same')(x)
        if filters>127:
            x = Conv2DTranspose(filters//2, (1,1), strides=(1, 1), padding='same')(x)
        x = Add()([x,connect])
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    def outBlock(inputs,filters=32,size=(3,3)):
        x = UpSampling2D(size=(2, 2))(inputs)
        x = Conv2DTranspose(filters, size, strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = keras.activations.tanh(x)
        return x
    inputs = Input((inpshapex,inpshapey, inpBands))
    
    block1 = convBlock(inputs,filters=32)
    block2 = convBlock(block1,filters=64)
    block3 = convBlock(block2,filters=128)
    block4 = convBlock(block3,filters=256)
    block5 = convBlock(block4,filters=512)
    
    
    
    block6= deconvBlock(block5,block4,filters=256)
    block7= deconvBlock(block6,block3,filters=128)
    block8= deconvBlock(block7,block2,filters=64)
    block9= deconvBlock(block8,block1,filters=32)
    
    outputs = outBlock(block9,filters=outBands)
    
    model = Model(inputs=[inputs], outputs=[outputs])

    return model
    