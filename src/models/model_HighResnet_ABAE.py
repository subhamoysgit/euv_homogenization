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
def make_CNN(reg='anc',features=32, rng=None):

	if rng is None:
		rng = np.random.default_rng()

	if optimizer is None:
		optimizer =  tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5)

##################initial layer###########################
	W_var_i = 15/((64+2)*(64+2)*2)
	W_lambda_i = 1/(2*W_var_i)
	W_anc_i = rng.normal(loc=0,scale=np.sqrt(W_var_i),size=[5,5,2,features])
	W_init_i = rng.normal(loc=0,scale=np.sqrt(W_var_i),size=[5,5,2,features])
	b_var_i = W_var_i
	b_lambda_i = 1/(2*b_var_i)
	b_anc_i = rng.normal(loc=0,scale=np.sqrt(b_var_i),size=[features])
	b_init_i = rng.normal(loc=0,scale=np.sqrt(b_var_i),size=[features])

	# create custom regulariser
	def custom_reg_W_i(weight_matrix):
		if reg == 'reg':
			return K.sum(K.square(weight_matrix)) * W_lambda_i/n_data
		elif reg == 'free':
			return 0.
		elif reg == 'anc':
			return K.sum(K.square(weight_matrix - W_anc_i)) * W_lambda_i/n_data

	def custom_reg_b_i(weight_matrix):
		if reg == 'reg':
			return K.sum(K.square(weight_matrix)) * b_lambda_i/n_data
		elif reg == 'free':
			return 0.
		elif reg == 'anc':
			return K.sum(K.square(weight_matrix - b_anc_i)) * b_lambda_i/n_data


################ middle layers ################## 
	W_var_m = 1/((64+2)*(64+2)*32)
	W_lambda_m = 1/(2*W_var_m)
	W_anc_m = rng.normal(loc=0,scale=np.sqrt(W_var_m),size=[5,5,features,features])
	W_init_m = rng.normal(loc=0,scale=np.sqrt(W_var_m),size=[5,5,features,features])
	b_var_m = W_var_m
	b_lambda_m = 1/(2*b_var_m)
	b_anc_m = rng.normal(loc=0,scale=np.sqrt(b_var_m),size=[features])
	b_init_m = rng.normal(loc=0,scale=np.sqrt(b_var_m),size=[features])
# create custom regulariser
	def custom_reg_W_m(weight_matrix):
		if reg == 'reg':
			return K.sum(K.square(weight_matrix)) * W_lambda_m/n_data
		elif reg == 'free':
			return 0.
		elif reg == 'anc':
			return K.sum(K.square(weight_matrix - W_anc_m)) * W_lambda_m/n_data

	def custom_reg_b_m(weight_matrix):
		if reg == 'reg':
			return K.sum(K.square(weight_matrix)) * b_lambda_m/n_data
		elif reg == 'free':
			return 0.
		elif reg == 'anc':
			return K.sum(K.square(weight_matrix - b_anc_m)) * b_lambda_m/n_data

########### upsampling layer ##########################
	W_var_u = 1/((4*64+2)*(4*64+2)*features)
	W_lambda_u = 1/(2*W_var_u)
	W_anc_u = rng.normal(loc=0,scale=np.sqrt(W_var_u),size=[5,5,features,features])
	W_init_u = rng.normal(loc=0,scale=np.sqrt(W_var_u),size=[5,5,features,features])
	b_var_u = W_var_u
	b_lambda_u = 1/(2*b_var_u)
	b_anc_u = rng.normal(loc=0,scale=np.sqrt(b_var_u),size=[features])
	b_init_u = rng.normal(loc=0,scale=np.sqrt(b_var_u),size=[features])

# create custom regulariser
	def custom_reg_W_u(weight_matrix):
		if reg == 'reg':
			return K.sum(K.square(weight_matrix)) * W_lambda_u/n_data
		elif reg == 'free':
			return 0.
		elif reg == 'anc':
			return K.sum(K.square(weight_matrix - W_anc_u)) * W_lambda_u/n_data

	def custom_reg_b_u(weight_matrix):
		if reg == 'reg':
			return K.sum(K.square(weight_matrix)) * b_lambda_u/n_data
		elif reg == 'free':
			return 0.
		elif reg == 'anc':
			return K.sum(K.square(weight_matrix - b_anc_u)) * b_lambda_u/n_data

######### final layer ###################################
	W_var_f = 5/((4*64)*(4*64)*32)
	W_lambda_f = 1/(2*W_var_f)
	W_anc_f = rng.normal(loc=0,scale=np.sqrt(W_var_f),size=[1,1,features,1])
	W_init_f = rng.normal(loc=0,scale=np.sqrt(W_var_f),size=[1,1,features,1])
	b_var_f = W_var_f
	b_lambda_f = 1/(2*b_var_f)
	b_anc_f = rng.normal(loc=0,scale=np.sqrt(b_var_f),size=[1])
	b_init_f = rng.normal(loc=0,scale=np.sqrt(b_var_f),size=[1])

# create custom regulariser
	def custom_reg_W_f(weight_matrix):
		if reg == 'reg':
			return K.sum(K.square(weight_matrix)) * W_lambda_f/n_data
		elif reg == 'free':
			return 0.
		elif reg == 'anc':
			return K.sum(K.square(weight_matrix - W_anc_f)) * W_lambda_f/n_data

	def custom_reg_b_f(weight_matrix):
		if reg == 'reg':
			return K.sum(K.square(weight_matrix)) * b_lambda_f/n_data
		elif reg == 'free':
			return 0.
		elif reg == 'anc':
			return K.sum(K.square(weight_matrix - b_anc_f)) * b_lambda_f/n_data



############# model architecture ################################

	def ResidualBlock(layer_before, features, activation = PReLU(), padding = 'valid'):
		padded = ReflectionPadding2D(padding=(2,2))(layer_before)
		conv = Conv2D(features, 5, activation = activation, padding = padding,kernel_initializer=initializers.Constant(value=W_init_m),bias_initializer=initializers.Constant(value=b_init_m),kernel_regularizer=custom_reg_W_m,bias_regularizer=custom_reg_b_m)(padded)
		padded = ReflectionPadding2D(padding=(2,2))(conv)
		conv = Conv2D(features, 5, activation = activation, padding = padding,kernel_initializer=initializers.Constant(value=W_init_m),bias_initializer=initializers.Constant(value=b_init_m),kernel_regularizer=custom_reg_W_m,bias_regularizer=custom_reg_b_m)(padded)
		return layer_before + conv


	def Encoder(layer_before,features):
		padded = ReflectionPadding2D(padding=(2,2))(layer_before)
		conv = Conv2D(features, 5, activation = PReLU(), padding = 'valid', kernel_initializer=initializers.Constant(value=W_init_i),bias_initializer=initializers.Constant(value=b_init_i),kernel_regularizer=custom_reg_W_i,bias_regularizer=custom_reg_b_i)(padded)
		for i in range(2):
			conv = ResidualBlock(conv, features, activation = PReLU(), padding = 'valid')
		padded = ReflectionPadding2D(padding=(2,2))(conv)
		conv = Conv2D(features, 5, activation = PReLU(), padding = 'valid', kernel_initializer=initializers.Constant(value=W_init_m),bias_initializer=initializers.Constant(value=b_init_m),kernel_regularizer=custom_reg_W_m,bias_regularizer=custom_reg_b_m)(padded)
		return conv


	def Decoder(layer_before,features):
		up = UpSampling2D(size = (4,4),interpolation = 'bilinear')(layer_before)
		padded = ReflectionPadding2D(padding=(2,2))(up)
		conv = Conv2D(features, 5, activation = PReLU(), padding = 'valid', kernel_initializer=initializers.Constant(value=W_init_u),bias_initializer=initializers.Constant(value=b_init_u),kernel_regularizer=custom_reg_W_u,bias_regularizer=custom_reg_b_u)(padded)
		conv = Conv2D(1, 1, activation = PReLU(), padding = 'valid', kernel_initializer=initializers.Constant(value=W_init_f),bias_initializer=initializers.Constant(value=b_init_f),kernel_regularizer=custom_reg_W_f,bias_regularizer=custom_reg_b_f)(conv)
		return conv


	inputs = Input(shape=(64,64,2))
	conv = Encoder(inputs, features)
	out = Decoder(conv, features)
	model = Model(inputs = inputs, outputs = out)
	
	return model
