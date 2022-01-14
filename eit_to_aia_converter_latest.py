#!/sr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:53:10 2020

@author: subhamoy
"""
import matplotlib.pyplot as plt
from scipy import ndimage
import tensorflow as tf
import numpy as np
import cv2
import sunpy.map
from sunpy.map import Map
import astropy.units as u
import keras
import numpy as np
import os
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from skimage import img_as_ubyte
from keras.models import *
from keras.layers import *
from keras.optimizers import *
seed_value = 5421
from keras.engine.topology import Layer
from keras.engine import InputSpec
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from keras.layers import PReLU
from degrid_new import degrid_new
def eit_to_aia_converter(file_e):

    class ReflectionPadding2D(Layer):
        def __init__(self, padding=(1, 1), **kwargs):
            self.padding = tuple(padding)
            self.input_spec = [InputSpec(ndim=4)]
            super(ReflectionPadding2D, self).__init__(**kwargs)

        def get_output_shape_for(self, s):
            """ If you are using "channels_last" configuration"""
            return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

        def call(self, x, mask=None):
            w_pad,h_pad = self.padding
            return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')


    from keras.layers import PReLU
    def ResidualBlock(layer_before, features, activation = PReLU(), padding = 'valid'):
        padded = ReflectionPadding2D(padding=(2,2))(layer_before)
        conv = Conv2D(features, 5, activation = activation, padding = padding)(padded)
        padded = ReflectionPadding2D(padding=(2,2))(conv)
        conv = Conv2D(features, 5, activation = activation, padding = padding)(padded)
        return layer_before + conv


    def Encoder(layer_before,features):
        padded = ReflectionPadding2D(padding=(2,2))(layer_before)
        conv = Conv2D(features, 5, activation = PReLU(), padding = 'valid')(padded)
        for i in range(2):
          conv = ResidualBlock(conv, features, activation = PReLU(), padding = 'valid')
        padded = ReflectionPadding2D(padding=(2,2))(conv)
        conv = Conv2D(features, 5, activation = PReLU(), padding = 'valid')(padded)
        return conv

    def Decoder(layer_before,features):
        up = UpSampling2D(size = (4,4),interpolation = 'bilinear')(layer_before)
        padded = ReflectionPadding2D(padding=(2,2))(up)
        conv = Conv2D(features, 5, activation = PReLU(), padding = 'valid')(padded)
        conv = Conv2D(1, 1, activation = PReLU(), padding = 'valid')(conv)
        return conv


    inputs = Input(shape=(64,64,2))
    conv = Encoder(inputs, 32)
    out = Decoder(conv, 32)
    model = Model(inputs = inputs, outputs = out)


    adam = Adam(lr=0.0001,beta_1=0.5)
    model.compile(optimizer=adam, loss = 'mean_squared_error', metrics=['mean_squared_error'])
    model.load_weights('/d0/models/eit_aia_sr_big_v3.h5')


    ####rescaled EIT map
    EITmap = Map(file_e)
    meta_e = EITmap.meta
    padX = [int(1024 - meta_e['crpix1']), int(meta_e['crpix1'])]
    padY = [int(1024 - meta_e['crpix2']), int(meta_e['crpix2'])]
    EIT_degrid = degrid_new(file_e)
    imgP = np.pad(EIT_degrid/(100*meta_e['exptime']), [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, meta_e['sc_roll'], reshape=False)
    imgC = imgR[padY[0] : -padY[1], padX[0] : -padX[1]]
    EIT_degrid = imgC
    eit_lr = Map(EIT_degrid,EITmap.meta)
    EITmap = eit_lr.rotate(recenter=True)
    x, y = np.meshgrid(*[np.arange(v.value) for v in EITmap.dimensions]) * u.pixel
    hpc_coords = EITmap.pixel_to_world(x, y)
    rSun = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / (EITmap.rsun_obs)
    prof = rSun
    eit_cnvrt = np.zeros((4096, 4096))
    patch_size = 64
    grid_size = 16    
    patches = np.zeros((grid_size**2,64,64,2))
    for i in range(grid_size**2):
        y0 = int(patch_size*(i % grid_size))
        x0 = int(patch_size*(i//grid_size)) 
        x1 = int(x0+patch_size)
        y1 = int(y0+patch_size)
        patches[i,:,:,0] = EITmap.data[y0:y1,x0:x1]
        patches[i,:,:,1]= prof[y0:y1,x0:x1]
    pred = model.predict(patches)
    for i in range(grid_size**2):
        y0 = int(patch_size*(i % grid_size))
        x0 = int(patch_size*(i//grid_size))
        x1 = int(x0+patch_size)
        y1 = int(y0+patch_size)
        eit_cnvrt[4*y0:4*y1,4*x0:4*x1] = pred[i,:,:,0]
    return eit_cnvrt,prof
    







