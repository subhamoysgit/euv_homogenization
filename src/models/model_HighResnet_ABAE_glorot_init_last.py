import os
import sys

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import initializers
from keras.layers import *
from keras.models import *

# Append source folder to system path
_MODEL_DIR = os.path.abspath(__file__).split('/')[:-1]
_SRC_DIR = os.path.join('/',*_MODEL_DIR[:-1])
sys.path.append(_SRC_DIR)

## Native Modules
from models.layer_2D_reflection_padding import ReflectionPadding2D

# CNN object
def make_CNN(reg='anc',
             features=32, 
			 rng=None, 
			 W_st_f=1.0, 
			 W_lambda_f=0.5, 
			 b_st_f=1.0,b_lambda_f=0.5):

	if rng is None:
		rng = np.random.default_rng()
	seed = int(rng.random()*1000)
	glorot = tf.keras.initializers.GlorotUniform(seed=seed)


######### final layer ###################################

	W_anc_f = W_st_f*glorot(shape=[1,1,features,1])
	b_anc_f = b_st_f*glorot(shape=[1])
	W_lambda_f = W_lambda_f*(features*1*1 + 1*1*1)/6
	b_lambda_f = b_lambda_f*(1)/6

# create custom regulariser
	def custom_reg_W_f(weight_matrix):
		if reg == 'reg':
			return K.sum(K.square(weight_matrix)) * W_lambda_f
		elif reg == 'free':
			return 0.
		elif reg == 'anc':
			return K.sum(K.square(weight_matrix - W_anc_f)) * W_lambda_f

	def custom_reg_b_f(weight_matrix):
		if reg == 'reg':
			return K.sum(K.square(weight_matrix)) * b_lambda_f
		elif reg == 'free':
			return 0.
		elif reg == 'anc':
			return K.sum(K.square(weight_matrix - b_anc_f)) * b_lambda_f



############# model architecture ################################

	def ResidualBlock(layer_before, features, activation = PReLU(), padding = 'valid'):
		padded = ReflectionPadding2D(padding=(2,2))(layer_before)
		conv = Conv2D(features, 5, activation = activation, padding = padding,kernel_initializer=glorot, bias_initializer=glorot)(padded)
		padded = ReflectionPadding2D(padding=(2,2))(conv)
		conv = Conv2D(features, 5, activation = activation, padding = padding,kernel_initializer=glorot, bias_initializer=glorot)(padded)
		return layer_before + conv


	def Encoder(layer_before,features):
		padded = ReflectionPadding2D(padding=(2,2))(layer_before)
		conv = Conv2D(features, 5, activation = PReLU(), padding = 'valid', kernel_initializer=glorot, bias_initializer=glorot)(padded)
		for i in range(2):
			conv = ResidualBlock(conv, features, activation = PReLU(), padding = 'valid')
		padded = ReflectionPadding2D(padding=(2,2))(conv)
		conv = Conv2D(features, 5, activation = PReLU(), padding = 'valid', kernel_initializer=glorot, bias_initializer=glorot)(padded)
		return conv


	def Decoder(layer_before,features):
		up = UpSampling2D(size = (4,4),interpolation = 'bilinear')(layer_before)
		padded = ReflectionPadding2D(padding=(2,2))(up)
		conv = Conv2D(features, 5, activation = PReLU(), padding = 'valid', kernel_initializer=glorot, bias_initializer=glorot)(padded)
		conv = Conv2D(1, 1, activation = PReLU(), padding = 'valid', kernel_initializer=glorot, bias_initializer=glorot, kernel_regularizer=custom_reg_W_f, bias_regularizer=custom_reg_b_f)(conv)
		return conv


	inputs = Input(shape=(64,64,2))
	conv = Encoder(inputs, features)
	out = Decoder(conv, features)
	model = Model(inputs = inputs, outputs = out)
	
	return model
