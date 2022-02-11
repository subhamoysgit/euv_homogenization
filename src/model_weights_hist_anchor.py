#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:53:10 2020

@author: subhamoy
"""
import os

import numpy as np
import sunpy.map

import tensorflow as tf
from keras import backend as K
import pickle
from keras.callbacks import ModelCheckpoint
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import initializers
from tensorflow.python.keras.layers import Layer, InputSpec
from keras.layers import PReLU
from keras.callbacks import ModelCheckpoint #EarlyStopping,LambdaCallback
import matplotlib.pyplot as plt
seed_value = 5421
rng = np.random.default_rng(seed_value)

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


############# model architecture ################################
features = 32
data_noise = 0.1
##################initial layer###########################
W_var_i = 0.1 #1/(5*5*2)
W_lambda_i = data_noise/W_var_i
W_anc_i = np.random.normal(loc=0,scale=np.sqrt(W_var_i),size=[5,5,2,features])
W_init_i = np.random.normal(loc=0,scale=np.sqrt(W_var_i),size=[5,5,2,features])
b_var_i = W_var_i
b_lambda_i = data_noise/b_var_i
b_anc_i = np.random.normal(loc=0,scale=np.sqrt(b_var_i),size=[features])
b_init_i = np.random.normal(loc=0,scale=np.sqrt(b_var_i),size=[features])


################ middle layers ################## 
W_var_m = 0.1 #1/(5*5*32)
W_lambda_m = data_noise/W_var_m
W_anc_m = np.random.normal(loc=0,scale=np.sqrt(W_var_m),size=[5,5,features,features])
W_init_m = np.random.normal(loc=0,scale=np.sqrt(W_var_m),size=[5,5,features,features])
b_var_m = W_var_m
b_lambda_m = data_noise/b_var_m
b_anc_m = np.random.normal(loc=0,scale=np.sqrt(b_var_m),size=[features])
b_init_m = np.random.normal(loc=0,scale=np.sqrt(b_var_m),size=[features])

########### upsampling layer ##########################
W_var_u = 0.1 #1/(5*5*features)
W_lambda_u = data_noise/W_var_u
W_anc_u = np.random.normal(loc=0,scale=np.sqrt(W_var_u),size=[5,5,features,features])
W_init_u = np.random.normal(loc=0,scale=np.sqrt(W_var_u),size=[5,5,features,features])
b_var_u = W_var_u
b_lambda_u = data_noise/b_var_u
b_anc_u = np.random.normal(loc=0,scale=np.sqrt(b_var_u),size=[features])
b_init_u = np.random.normal(loc=0,scale=np.sqrt(b_var_u),size=[features])

######### final layer ###################################
W_var_f = 0.1 #1/(5*5*features)
W_lambda_f = data_noise/W_var_f
W_anc_f = np.random.normal(loc=0,scale=np.sqrt(W_var_f),size=[1,1,features,1])
W_init_f = np.random.normal(loc=0,scale=np.sqrt(W_var_f),size=[1,1,features,1])
b_var_f = W_var_f
b_lambda_f = data_noise/b_var_f
b_anc_f = np.random.normal(loc=0,scale=np.sqrt(b_var_f),size=[1])
b_init_f = np.random.normal(loc=0,scale=np.sqrt(b_var_f),size=[1])




def ResidualBlock(layer_before, features, activation = PReLU(), padding = 'valid'):
	padded = ReflectionPadding2D(padding=(2,2))(layer_before)
	conv = Conv2D(features, 5, activation = activation, padding = padding,kernel_initializer=initializers.Constant(value=W_init_m),bias_initializer=initializers.Constant(value=b_init_m))(padded)
	padded = ReflectionPadding2D(padding=(2,2))(conv)
	conv = Conv2D(features, 5, activation = activation, padding = padding,kernel_initializer=initializers.Constant(value=W_init_m),bias_initializer=initializers.Constant(value=b_init_m))(padded)
	return layer_before + conv


def Encoder(layer_before,features):
	padded = ReflectionPadding2D(padding=(2,2))(layer_before)
	conv = Conv2D(features, 5, activation = PReLU(), padding = 'valid', kernel_initializer=initializers.Constant(value=W_init_i),bias_initializer=initializers.Constant(value=b_init_i))(padded)
	for i in range(2):
		conv = ResidualBlock(conv, features, activation = PReLU(), padding = 'valid')
	padded = ReflectionPadding2D(padding=(2,2))(conv)
	conv = Conv2D(features, 5, activation = PReLU(), padding = 'valid', kernel_initializer=initializers.Constant(value=W_init_m),bias_initializer=initializers.Constant(value=b_init_m))(padded)
	return conv


def Decoder(layer_before,features):
	up = UpSampling2D(size = (4,4),interpolation = 'bilinear')(layer_before)
	padded = ReflectionPadding2D(padding=(2,2))(up)
	conv = Conv2D(features, 5, activation = PReLU(), padding = 'valid', kernel_initializer=initializers.Constant(value=W_init_u),bias_initializer=initializers.Constant(value=b_init_u))(padded)
	conv = Conv2D(1, 1, activation = PReLU(), padding = 'valid', kernel_initializer=initializers.Constant(value=W_init_f),bias_initializer=initializers.Constant(value=b_init_f))(conv)
	return conv


inputs = Input(shape=(64,64,2))
conv = Encoder(inputs, features)
out = Decoder(conv, features)
model = Model(inputs = inputs, outputs = out)

#### before training ###
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
for layer in model.layers:
	weights = layer.get_weights()
	if weights:  ###for layers with weights
		print(weights[0].shape)
		plt.hist(weights[0].flatten(),bins = 20,alpha = 0.2,label=layer.name)
		plt.yscale('log')
plt.title('Before Training')
plt.legend(frameon=False)	

#### after training ###
model.load_weights("/d0/models/eit_aia_sr_big_ancabae01.h5")
plt.subplot(1,2,2)
for layer in model.layers:
	weights = layer.get_weights()
	if weights:  ###for layers with weights
		plt.hist(weights[0].flatten(),bins = 20,alpha = 0.2,label=layer.name)
		plt.yscale('log')
plt.title('After Training')
plt.legend(frameon=False)	
plt.show()