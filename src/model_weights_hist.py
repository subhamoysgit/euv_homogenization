#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:53:10 2020

@author: subhamoy
"""
import os

import numpy as np
from sklearn.metrics import mean_squared_error
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
conv = Encoder(inputs, features)
out = Decoder(conv, features)
model = Model(inputs = inputs, outputs = out)

#sgd = SGD(lr=0.0001, momentum=0.9, nesterov=True)
adam = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5)
model.compile(optimizer=adam, loss = mean_squared_error, metrics=[mean_squared_error],run_eagerly=True)

idx = 0
layer = model.layers[idx]
weights = layer.get_weights
w = weights[0]
print(w.shape)

