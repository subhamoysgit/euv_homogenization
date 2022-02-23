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

from keras.callbacks import ModelCheckpoint
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.layers import PReLU
from keras.callbacks import ModelCheckpoint

## Loss and loss term weigths
from losses.combined_loss_ssim_grad_hist import combined_loss

COEF_SSIM = 1   # weight of the SSIM term
COEF_GRAD = 4   # weight of the gradient term
COEF_HIST = 0.2 # weight of the histogram term


## Data loader and data paths
from data_loaders.eit_aia_loader import imageIndexer, imageLoader
PATCH_PATH = '/d0/patches/val/'
TRAIN_PATH = '/d1/patches/trn/'
VAL_PATH = '/d0/patches/val/'

## CNN, training run, and ensemble definitions
from models.model_HighResnet_ABAE import make_CNN
BS = 1
ENSEMBLE_N = 10	 # no. CNNs in ensemble
REGULARIZATION = 'anc'

## Constants that control the run
SEED_VALUE = 5421
rng = np.random.default_rng(SEED_VALUE)










# bs = 1
# TRAIN_PATH = '/d1/patches/trn/'
# VAL_PATH = '/d0/patches/val/'
# FD_PATH_TRN = '/d1/fd/trn/eit_171/'
# fd_path_val = '/d1/fd/val/eit_171/'
# epoch = 1 #intial epoch
# L = 1536*(196 + 169)//bs 
# L1 = 451*(196 + 169)//bs

# # CNN options
# ENSEMBLE_N = 10	# no. CNNs in ensemble
# reg = 'anc'		# type of regularisation to use - anc (anchoring) reg (regularised) free (unconstrained)

# # data options
# n_data = 4*(L + L1)*bs 	# no. training + val data points
# seed_in = 0 # random seed used to produce data blobs - try changing to see how results look w different data


if __name__ == "__main__":

	# index training patches
	patch_num = imageIndexer(PATCH_PATH)

	# create the NNs
	CNNs=[]
	for m in range(ENSEMBLE_N):
		CNNs.append(make_CNN(reg=REGULARIZATION))
	print(CNNs[-1].summary())


	for m in range(ENSEMBLE_N):
		print('-- training: ' + str(m+1) + ' of ' + str(ENSEMBLE_N) + ' CNNs --') 
		checkpoint = ModelCheckpoint("/d0/models/eit_aia_sr_big_abae"+str(m+1).zfill(2)+".h5", monitor='val_combined_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
		history = CNNs[m].fit(imageLoader(TRAIN_PATH, BS, patch_num), batch_size = 4*BS, steps_per_epoch = L, epochs = 10,callbacks=[checkpoint], validation_data=imageLoader(VAL_PATH, BS, patch_num), validation_steps = L1,initial_epoch=epoch-1)
