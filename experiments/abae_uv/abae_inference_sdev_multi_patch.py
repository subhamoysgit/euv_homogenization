#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
----------------------------------------------------------------------------------------------
TEMPLATE DESCRIPTION
----------------------------------------------------------------------------------------------


@author: Subhamoy Chatterjee
@author: Andres Munoz-Jaramillo

Template file to train single frame Super resolution.  It requires a combined loss function from the src/losses folder, a model from the
/src/models folder, and a loader from the /src/loaders folder.

- Copy this file inside any prefered folder structure inside the /src/experiments folder and modify there
- Make sure the appending of the src folder is done properly for it to find native modules (see below)
- Add a date and clear description of your experiment.  Bonus points for logging results afterwards :)
- Profit (hopefully)

"""

"""
----------------------------------------------------------------------------------------------
EXPERIMENT DESCRIPTION
----------------------------------------------------------------------------------------------
Feb 23, 2022

loss is MSE

First run abae memorizing a samll subset of the data. See if we can memorize, and see how does
ensemble behave with memorized vs. unseen data


----------------------------------------------------------------------------------------------
RESULTS
----------------------------------------------------------------------------------------------


What was the outcome of the experiment?  How did it match your expectatios and hopes?
Is this your best model to date?

"""


# System paths and GPU vs. CPU
import os
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Append source folder to system path.  It uses the folder where the experiment runs.
# Since the template file is in 'src/templates' you only need to remove the last folder i.e. you only need '-1' in
# os.path.abspath(__file__).split('/')[:-1].  If you make your folder structure deeper, be sure to increase this value.
 
_MODEL_DIR = os.path.abspath(__file__).split('/')[:-2]
_SRC_DIR = os.path.join('/',*_MODEL_DIR[:-1])
_SRC_DIR = os.path.join(_SRC_DIR,'src')

sys.path.append(_SRC_DIR)


# Load modules
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import pickle
import matplotlib.pyplot as plt
##------------------------------------------------------------------------------------
## Random seed initialization
SEED_VALUE = 42
rng = np.random.default_rng(SEED_VALUE)


##------------------------------------------------------------------------------------
## Load CNN model and CNN coefficients
from models.model_HighResnet_ABAE_normal_init import make_CNN

# CNN options
ENSEMBLE_SIZE = 4  # no. CNNs in ensemble
REGULARIZATION = 'anc'  # type of regularisation to use - anc (anchoring) reg (regularised) free (unconstrained)
BATCH_SIZE = 10  # Batch Size
EPOCH0 = 1  # First epoch

DATA_NOISE = 0.1 # noise variance as mean of aia patch hist
W_VAR_I = 0.1 # variance of the anchor weights
W_LAMBDA_I = 0.000001 # Strength of the regularization term for anchor weights
B_VAR_I = W_VAR_I # variance of the anchor biases 
B_LAMBDA_I = W_LAMBDA_I # Strength of the regularization term for anchor biases


##------------------------------------------------------------------------------------
## Patch location, data loader, and augmentation
from data_loaders.eit_aia_loader import imageIndexer, imageLoader

TRAIN_PATH = '/d1/patches/trn/'  # Training data path


# Augmentation
VFLIP = True  # Vertical flip
HFLIP = True  # Horizontal flip

##------------------------------------------------------------------------------------
## Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5)


OUTPUT_FOLDER = '/d0/models/'
OUTPUT_FILE = 'eit_aia_sr_abae_medium_GLOROT_UNIF_last_LAMBDA_SCALED_001_ST_2_'#'eit_aia_sr_big_v17'#'eit_aia_sr_abae_small_LAMBDA_01_VAR_1_'
ANCHOR_PARAMS  = '_GLOROT_UNIF_last_LAMBDA_SCALED_001_ST_2_'
TRAIN_DATE_RANGE = [20140101,20141231]
VAL_DATE_RANGE = [20160101,20160229]
TEST_DATE_RANGE = [20170101,20170229]

EIT_TRN = []
for root,dirs,files in os.walk(TRAIN_PATH):
	for file in files:
		if file[:3]=='eit':
			if TRAIN_DATE_RANGE:
				if int(file[4:12])>=TRAIN_DATE_RANGE[0] and int(file[4:12])<=TRAIN_DATE_RANGE[1]:
					EIT_TRN.append(root+file)
			else:
				EIT_TRN.append(root+file)


EIT_VAL = []
for root,dirs,files in os.walk(TRAIN_PATH):
	for file in files:
		if file[:3]=='eit':
			if VAL_DATE_RANGE:
				if int(file[4:12])>=VAL_DATE_RANGE[0] and int(file[4:12])<=VAL_DATE_RANGE[1]:
					EIT_VAL.append(root+file)
			else:
				EIT_VAL.append(root+file)

EIT_TST = []
for root,dirs,files in os.walk(TRAIN_PATH):
	for file in files:
		if file[:3]=='eit':
			if TEST_DATE_RANGE:
				if int(file[4:12])>=TEST_DATE_RANGE[0] and int(file[4:12])<=TEST_DATE_RANGE[1]:
					EIT_TST.append(root+file)
			else:
				EIT_TST.append(root+file)


EIT_TRN = sorted(EIT_TRN)
EIT_VAL = sorted(EIT_VAL)
EIT_TST = sorted(EIT_TST)
import random
random.Random(203).shuffle(EIT_TRN)
random.Random(203).shuffle(EIT_VAL)
random.Random(203).shuffle(EIT_TST)
N_PATCHES = 1000
if __name__ == "__main__":
	best_epochs = np.zeros(ENSEMBLE_SIZE)
	ARRAY_TRN = np.zeros((N_PATCHES*64*64,2))
	ARRAY_VAL = np.zeros((N_PATCHES*64*64,2))
	ARRAY_TST = np.zeros((N_PATCHES*64*64,2))


	for m in range(ENSEMBLE_SIZE):
		p = pickle.load(open(OUTPUT_FOLDER + OUTPUT_FILE + str(m+1).zfill(2) +'_'+'val_mse.p','rb'))
		best_epochs[m] = 1 + np.argmin(p)
	for i in range(N_PATCHES):
		PATCH_NAME = EIT_TRN[i]
		eit_tr = pickle.load(open(PATCH_NAME, "rb" ))
		PATCH_NAME = EIT_VAL[i]
		eit_vl = pickle.load(open(PATCH_NAME, "rb" ))
		PATCH_NAME = EIT_TST[i]
		eit_ts = pickle.load(open(PATCH_NAME, "rb" ))

		X_tr = np.zeros((1,64,64,2))
		prof = eit_tr[:,:,1]
		X_tr[0,:,:,0] = eit_tr[:,:,0]
		X_tr[0,:,:,1] = prof[:,:]

		X_vl = np.zeros((1,64,64,2))
		prof = eit_vl[:,:,1]
		X_vl[0,:,:,0] = eit_vl[:,:,0]
		X_vl[0,:,:,1] = prof[:,:]

		X_ts = np.zeros((1,64,64,2))
		prof = eit_ts[:,:,1]
		X_ts[0,:,:,0] = eit_ts[:,:,0]
		X_ts[0,:,:,1] = prof[:,:]

		# create the NNs
		CNNs=[]
		out_TRN = np.zeros((ENSEMBLE_SIZE,256,256))
		out_VAL = np.zeros((ENSEMBLE_SIZE,256,256))
		out_TST = np.zeros((ENSEMBLE_SIZE,256,256))
		for m in range(ENSEMBLE_SIZE):
			CNNs.append(make_CNN(reg=REGULARIZATION, features=32, rng=rng, W_var_i=W_VAR_I, W_lambda_i=W_LAMBDA_I, b_var_i=B_VAR_I, b_lambda_i=B_LAMBDA_I))
			CNNs[m].load_weights(OUTPUT_FOLDER + OUTPUT_FILE + str(m+1).zfill(2) +'_'+str(int(best_epochs[m])).zfill(2)+'.h5')
			pred = CNNs[m].predict(X_tr)
			out_TRN[m,:,:] = pred[0,:,:,0]
			pred = CNNs[m].predict(X_vl)
			out_VAL[m,:,:] = pred[0,:,:,0]
			pred = CNNs[m].predict(X_ts)
			out_TST[m,:,:] = pred[0,:,:,0]
			
		input_intensity_tr,input_intensity_vl,input_intensity_ts = [],[],[]
		output_sdev_tr,output_sdev_vl,output_sdev_ts = [],[],[]
		for ii in range(64):
			for jj in range(64):
				o = np.std(out_TRN[:,4*ii:4*(ii+1),4*jj:4*(jj+1)])
				input_intensity_tr.append(X_tr[0,ii,jj,0])
				output_sdev_tr.append(o)
				o = np.std(out_VAL[:,4*ii:4*(ii+1),4*jj:4*(jj+1)])
				input_intensity_vl.append(X_vl[0,ii,jj,0])
				output_sdev_vl.append(o)
				o = np.std(out_TST[:,4*ii:4*(ii+1),4*jj:4*(jj+1)])
				input_intensity_ts.append(X_ts[0,ii,jj,0])
				output_sdev_ts.append(o)
		ARRAY_TRN[i*64*64:(i+1)*64*64,0] = np.array(input_intensity_tr)
		ARRAY_TRN[i*64*64:(i+1)*64*64,1] = np.array(output_sdev_tr)
		ARRAY_VAL[i*64*64:(i+1)*64*64,0] = np.array(input_intensity_vl)
		ARRAY_VAL[i*64*64:(i+1)*64*64,1] = np.array(output_sdev_vl)
		ARRAY_TST[i*64*64:(i+1)*64*64,0] = np.array(input_intensity_ts)
		ARRAY_TST[i*64*64:(i+1)*64*64,1] = np.array(output_sdev_ts)
		pickle.dump(ARRAY_TRN,open('/d0/models/EUV_TRN_SDEV_70k_'+ANCHOR_PARAMS+str(N_PATCHES),'wb'))
		pickle.dump(ARRAY_VAL,open('/d0/models/EUV_VAL_SDEV_70k_'+ANCHOR_PARAMS+str(N_PATCHES),'wb'))
		pickle.dump(ARRAY_TST,open('/d0/models/EUV_TST_SDEV_70k_'+ANCHOR_PARAMS+str(N_PATCHES),'wb'))
		print(str(1 + i) + ' patches saved')