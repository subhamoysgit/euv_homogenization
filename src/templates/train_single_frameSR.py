#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:53:10 2020

@author: subhamoy
"""
# System paths and GPU vs. CPU
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Append source folder to system path.  It uses the folder where the experiment runs.
# Since the template file is in 'src/templates' you only need to remove the last folder i.e. you only need '-1' in
# os.path.abspath(__file__).split('/')[:-1].  If you make your folder structure deeper, be sure to increase this value.
 
_MODEL_DIR = os.path.abspath(__file__).split('/')[:-1]
_SRC_DIR = os.path.join('/',*_MODEL_DIR[:-1])
sys.path.append(_SRC_DIR)

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

## Native losses
from losses.combined_loss_ssim_grad_hist import combined_loss

##------------------------------------------------------------------------------------
## Constants that control the run
SEED_VALUE = 5421
rng = np.random.default_rng(SEED_VALUE)

## Batches and Epochs
PATCH_PATH = '/d0/patches/val/'

## Loss term weigths
COEF_SSIM = 1
COEF_GRAD = 4
COEF_HIST = 0.2


patch_num = []
for root,dirs,files in os.walk(PATCH_PATH):
	for file in files:
			if file.startswith('eit') and file[4:12]==f[3:11]:
				patches_e = pickle.load(open(root+file, "rb" ))
				if np.sum(patches_e[:,:,1]<1)>=0:#.2*64*64:
					patch_num.append(file[13:18])


def imageLoader(file_path, batch_size,patch_num,fd_path):
	files_e = []
	files_d = []
	for root,dirs,files in os.walk(fd_path):
		for file in files:
				files_e.append(file)
				files_d.append(file[3:11])

	file_list = []
	for root,dirs,files in os.walk(file_path):
		for file in files:
			if file[:3]=='eit' and file[13:18] in patch_num:
				file_list.append(root+file)
	file_list = file_list[:batch_size*(len(file_list)//batch_size)]

	rng.shuffle(file_list)
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
			aia = pickle.load(open(file_list[k*batch_size+i][:16]+'aia'+file_list[k*batch_size+i][19:], "rb" ))
			X[i,:,:,0] = eit[:,:,0]*1000*e.meta['exptime']
			prof = eit[:,:,1]
			X[i,:,:,1] = prof[:,:]
			Y[i,:,:,0] = aia[:,:]
		X[batch_size:2*batch_size,:,:,:] = np.flip(X[:batch_size,:,:,:].copy(),1)
		X[2*batch_size:3*batch_size,:,:,:] = np.flip(X[:batch_size,:,:,:].copy(),2)
		X[3*batch_size:4*batch_size,:,:,:] = np.flip(np.flip(X[:batch_size,:,:,:].copy(),1),2)
		Y[batch_size:2*batch_size,:,:,0] = np.flip(Y[:batch_size,:,:,0].copy(),1)
		Y[2*batch_size:3*batch_size,:,:,0] = np.flip(Y[:batch_size,:,:,0].copy(),2)
		Y[3*batch_size:4*batch_size,:,:,0] = np.flip(np.flip(Y[:batch_size,:,:,0].copy(),1),2)
		k = k+1
		if k == num:
			rng.shuffle(file_list)

		yield(X,Y)


bs = 1
train_path = '/d1/patches/trn/'
val_path = '/d0/patches/val/'
fd_path_trn = '/d1/fd/trn/eit_171/'
fd_path_val = '/d1/fd/val/eit_171/'
epoch = 1 #intial epoch
L = 1536*(196 + 169)//bs 
L1 = 451*(196 + 169)//bs

# CNN options
n_ensemble = 10	# no. CNNs in ensemble
reg = 'anc'		# type of regularisation to use - anc (anchoring) reg (regularised) free (unconstrained)

# data options
n_data = 4*(L + L1)*bs 	# no. training + val data points
seed_in = 0 # random seed used to produce data blobs - try changing to see how results look w different data

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





# CNN object
def fn_make_CNN(reg='anc',features=32):
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

	#sgd = SGD(lr=0.0001, momentum=0.9, nesterov=True)
	adam = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5)
	model.compile(optimizer=adam, loss = combined_loss(COEF_SSIM, COEF_GRAD, COEF_HIST), metrics=[combined_loss(COEF_SSIM, COEF_GRAD, COEF_HIST)],run_eagerly=True)
	
	return model



if __name__ == "__main__":

	# create the NNs
	CNNs=[]
	for m in range(n_ensemble):
		CNNs.append(fn_make_CNN(reg=reg))
	print(CNNs[-1].summary())


	for m in range(n_ensemble):
		print('-- training: ' + str(m+1) + ' of ' + str(n_ensemble) + ' CNNs --') 
		checkpoint = ModelCheckpoint("/d0/models/eit_aia_sr_big_abae"+str(m+1).zfill(2)+".h5", monitor='val_combined_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
		#early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')#, baseline=None, restore_best_weights=True
		history=CNNs[m].fit(imageLoader(train_path, bs,patch_num,fd_path_trn), batch_size = 4*bs, steps_per_epoch = L, epochs = 10,callbacks=[checkpoint], validation_data=imageLoader(val_path, bs,patch_num,fd_path_val), validation_steps = L1,initial_epoch=epoch-1)
