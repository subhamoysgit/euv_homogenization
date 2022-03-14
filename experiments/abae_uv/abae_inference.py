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
ENSEMBLE_SIZE = 10  # no. CNNs in ensemble
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
VAL_PATH = '/d1/patches/trn/'  # Validation data path


# Augmentation
VFLIP = True  # Vertical flip
HFLIP = True  # Horizontal flip

##------------------------------------------------------------------------------------
## Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5)


OUTPUT_FOLDER = '/d0/models/'
OUTPUT_FILE = 'eit_aia_sr_abae_small_LAMBDA_0_VAR_d1_'#'eit_aia_sr_big_v17'#'eit_aia_sr_abae_small_LAMBDA_01_VAR_1_'
TRAIN_DATE_RANGE = [20140101,20141231]
VAL_DATE_RANGE = [20160101,20160229]

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
for root,dirs,files in os.walk(VAL_PATH):
	for file in files:
		if file[:3]=='eit':
			if VAL_DATE_RANGE:
				if int(file[4:12])>=VAL_DATE_RANGE[0] and int(file[4:12])<=VAL_DATE_RANGE[1]:
					EIT_VAL.append(root+file)
			else:
				EIT_VAL.append(root+file)


EIT_TRN = sorted(EIT_TRN)
EIT_VAL = sorted(EIT_VAL)

if __name__ == "__main__":
	PATCH_NAME = EIT_VAL[50]
	print(PATCH_NAME)
	eit = pickle.load(open(PATCH_NAME, "rb" ))
	aia = pickle.load(open(PATCH_NAME[:16]+'aia'+PATCH_NAME[19:], "rb" ))
	X = np.zeros((1,64,64,2))
	Y = np.zeros((1,256,256,1))
	prof = eit[:,:,1]
	X[0,:,:,0] = eit[:,:,0]
	X[0,:,:,1] = prof[:,:]
	Y[0,:,:,0] = aia[:,:]
	nTrain, nVal = imageIndexer(TRAIN_PATH, VAL_PATH, trainDateRange = TRAIN_DATE_RANGE, valDateRange = VAL_DATE_RANGE)
	print(nTrain)
	print(nVal)
	# create the NNs
	CNNs=[]
	AVAILABLE_ANCHORS = [1,2,3,4]
	l = 2+len(AVAILABLE_ANCHORS)
	fig,ax = plt.subplots(l,10)
	ax = ax.ravel()

	#for m in range(ENSEMBLE_SIZE):
	for m in range(ENSEMBLE_SIZE):
		CNNs.append(make_CNN(reg=REGULARIZATION, features=32, rng=rng, W_var_i=W_VAR_I, W_lambda_i=W_LAMBDA_I, b_var_i=B_VAR_I, b_lambda_i=B_LAMBDA_I))
		#CNNs[m].compile(optimizer=optimizer, loss = 'mse', metrics=['mse'], run_eagerly=True)
		
	for e in range(10):
		k = 0
		ax[e].imshow(X[0,:,:,0])
		ax[e].set_title('ep = '+str(e+1))
		ax[e].set_xticks([])
		ax[e].set_yticks([])
		ax[e + 10].imshow(Y[0,:,:,0],vmin = np.min(aia),vmax =np.max(aia))
		ax[e + 10].set_xticks([])
		ax[e + 10].set_yticks([])
		if e == 0:
			ax[e].set_ylabel('INPUT')
			ax[e + 10].set_ylabel('TARGET')
		for m in AVAILABLE_ANCHORS:
			CNNs[m-1].load_weights(OUTPUT_FOLDER + OUTPUT_FILE + str(m).zfill(2) +'_'+str(e+1).zfill(2)+'.h5')
			#CNNs[m-1].load_weights(OUTPUT_FOLDER + OUTPUT_FILE+'.h5')
			p = CNNs[m-1].predict(X)
			ax[e+10*(k+2)].imshow(p[0,:,:,0],vmin = np.min(aia),vmax =np.max(aia))
			ax[e+10*(k+2)].set_xticks([])
			ax[e+10*(k+2)].set_yticks([])
			if e == 0:
				ax[e+10*(k+2)].set_ylabel('ANC '+str(m))
			k = k + 1
	plt.show()