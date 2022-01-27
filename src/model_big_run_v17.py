#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:53:10 2020

@author: subhamoy
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from keras import backend as K
import pickle
from keras.callbacks import ModelCheckpoint
from keras.models import *
from keras.layers import *
from keras.optimizers import *
seed_value = 5421
np.random.seed(seed_value)


import os

fd_path = '/d1/fd/val/eit_171/'
patch_path = '/d0/patches/val/'
files_e = []
for root,dirs,files in os.walk(fd_path):
    for file in files:
        files_e.append(file)


file_list_eit = sorted(files_e)
f = file_list_eit[100]
patch_num = []
for root,dirs,files in os.walk(patch_path):
    for file in files:
        if file.startswith('eit') and file[4:12]==f[3:11]:
           patches_e = pickle.load(open(root+file, "rb" ))
           if np.sum(patches_e[:,:,1]<1)>=0:#.2*64*64:
              patch_num.append(file[13:18])

print(patch_num)



def imageLoader(file_path, batch_size,patch_num,fd_path):
  files_e = []
  files_d = []
  import os
  import sunpy.map
  for root,dirs,files in os.walk(fd_path):
      for file in files:
          files_e.append(file)
          files_d.append(file[3:11])
  import math
  import random
  #seed = 2390
  #random.seed(seed)
  file_list = []
  import os
  for root,dirs,files in os.walk(file_path):
      for file in files:
        if file[:3]=='eit' and file[13:18] in patch_num:
          file_list.append(root+file)
  file_list = file_list[:batch_size*(len(file_list)//batch_size)]
  print(len(file_list))
  #file_list = file_list[:10000]
  random.shuffle(file_list)
  k = 0
  num = len(file_list)//batch_size
  X = np.zeros((4*batch_size,64,64,2),dtype=float)
  Y = np.zeros((4*batch_size,256,256,1),dtype=float)
  while True:
    k = k % num
    for i in range(batch_size):
      idx = files_d.index(file_list[k*batch_size+i][20:28]) 
      e = sunpy.map.Map(fd_path+files_e[idx])
      eit = pickle.load(open(file_list[k*batch_size+i], "rb" ))
      #print([file_list[k*batch_size+i],'aia'+file_list[k*batch_size+i][3:]])
      aia = pickle.load(open(file_list[k*batch_size+i][:16]+'aia'+file_list[k*batch_size+i][19:], "rb" ))
      X[i,:,:,0] = eit[:,:,0]*1000*e.meta['exptime']
      prof = eit[:,:,1]
      #prof[prof>1] = 0
      #ind = np.where(prof<=1)
      #prof[ind] = 1 - prof[ind]**2 
      X[i,:,:,1] = prof[:,:]
      Y[i,:,:,0] = aia[:,:]
    X[batch_size:2*batch_size,:,:,:] = np.flip(X[:batch_size,:,:,:].copy(),1)
    X[2*batch_size:3*batch_size,:,:,:] = np.flip(X[:batch_size,:,:,:].copy(),2)
    X[3*batch_size:4*batch_size,:,:,:] = np.flip(np.flip(X[:batch_size,:,:,:].copy(),1),2)
    Y[batch_size:2*batch_size,:,:,0] = np.flip(Y[:batch_size,:,:,0].copy(),1)
    Y[2*batch_size:3*batch_size,:,:,0] = np.flip(Y[:batch_size,:,:,0].copy(),2)
    Y[3*batch_size:4*batch_size,:,:,0] = np.flip(np.flip(Y[:batch_size,:,:,0].copy(),1),2)
    k = k+1
    #print(k)
    if k == num:
      #seed = 2*seed
      #random.seed(seed)
      random.shuffle(file_list)
    # if k%10==0 and k>0:
    #   model.save_weights('/content/drive/My Drive/Colab Notebooks/swri_research/aia_eit_data_download/eit_aia_sr_v2.h5')

    yield(X,Y)



from tensorflow.python.keras.layers import Layer, InputSpec

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



def histogramGen(x):
  #plt.imshow(x[0,:,:,0])
  # print(shapeOp) #Tensor("Shape:0", shape=(2,), dtype=int32)
  # with tf.Session() as sess:
  #   print(sess.run(shapeOp))
  # if len(x.shape) == 3:
  #   x = x[None,:, :, :]
  # else:
  #   x = x[0, :, :, :]
  #   x = x[None,:]
  x = x*1000
  patch_size = x.shape[2]
  dl = 0.1
  lim = 3000 
  normalisation = 1 
  noise_level = 20
  lim = np.log10(lim)  # Maximum value

  bins = np.round(np.power(10, np.arange(1, lim + dl, dl)), 2)
  bins = bins - 10 + noise_level
  # Defining centers and bin widths for the learnable histogram
  centers = (bins[1:] + bins[0:-1]) / 2
  widths = (bins[1:] - bins[0:-1])
  nonzero = np.ones(centers.shape[0])
  weight1 = nonzero[None, None, None,:]
  bias1 = -centers * nonzero
  diag = -nonzero / widths
  weight2 = np.expand_dims(np.expand_dims(np.diag(diag), axis=0), axis=0)
  bias2 = nonzero
  conv = Conv2D(centers.shape[0], 1, weights= [weight1, bias1])(x)
  conv = K.abs(conv)
  conv = Conv2D(centers.shape[0], 1, weights= [weight2, bias2])(conv)
  conv = Activation('relu')(conv)
  hist = AveragePooling2D(pool_size=(patch_size, patch_size))(conv)
  #print(patch_size)
  return hist



def grad_x(x):
  import tensorflow as tf
  dx, dy = tf.image.image_gradients(x)
  return dx

def grad_y(x):
  import tensorflow as tf
  dx, dy = tf.image.image_gradients(x)
  return dy



def combined_loss(y_true,y_pred):
  return tf.math.reduce_mean(tf.square(y_true-y_pred)) - 0.001*tf.image.ssim(y_true, y_pred, max_val=10.0)
  #return 0.2*tf.math.reduce_mean(tf.square(histogramGen(y_true)-histogramGen(y_pred))) + tf.math.reduce_mean(tf.square(y_true-y_pred)) - tf.image.ssim(y_true, y_pred, max_val=10.0)#+ 4*tf.math.reduce_mean(tf.square(grad_x(y_true) - grad_x(y_pred))) + 4*tf.math.reduce_mean(tf.square(grad_y(y_true) - grad_y(y_pred)))





#sgd = SGD(lr=0.0001, momentum=0.9, nesterov=True)
adam = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5)
model.compile(optimizer=adam, loss = combined_loss, metrics=[combined_loss],run_eagerly=True)


bs = 10
train_path = '/d1/patches/trn/'
val_path = '/d0/patches/val/'
fd_path_trn = '/d1/fd/trn/eit_171/'
fd_path_val = '/d1/fd/val/eit_171/'
# file_e="/content/drive/My Drive/Colab Notebooks/swri_research/aia_eit_data_download/val/eit_20120803_0_39.p"
# file_a="/content/drive/My Drive/Colab Notebooks/swri_research/aia_eit_data_download/val/aia_20120803_0_39.p"
#mini_epoch = 5
epoch = 1
L = 1536*(196 + 169)//bs 
L1 = 451*(196 + 169)//bs

from keras.callbacks import ModelCheckpoint, EarlyStopping#,LambdaCallback
# cb_redraw = get_redraw(file_e,file_a,model)
# redraw_callback = LambdaCallback(on_batch_end=cb_redraw)

#model.load_weights("/d0/models/eit_aia_sr_big_v7.h5")
checkpoint = ModelCheckpoint("/d0/models/eit_aia_sr_big_v17.h5", monitor='val_combined_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
#early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')#, baseline=None, restore_best_weights=True
history=model.fit(imageLoader(train_path, bs,patch_num,fd_path_trn), batch_size = 4*bs, steps_per_epoch = L, epochs = 10,callbacks=[checkpoint], validation_data=imageLoader(val_path, bs,patch_num,fd_path_val), validation_steps = L1,initial_epoch=epoch-1)



