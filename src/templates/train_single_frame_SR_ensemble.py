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

Why are you doing this experiment?  What do you hope to see?  What changed from last time?



----------------------------------------------------------------------------------------------
RESULTS
----------------------------------------------------------------------------------------------

What was the outcome of the experiment?  How did it match your expectatios and hopes?
Is this your best model to date?

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


# Load modules
import numpy as np
import pickle
import tensorflow as tf
from keras.callbacks import ModelCheckpoint


##------------------------------------------------------------------------------------
## Random seed initialization
SEED_VALUE = 5421
rng = np.random.default_rng(SEED_VALUE)


##------------------------------------------------------------------------------------
## Load CNN model and CNN coefficients
from models.abae_high_resnet import fn_make_CNN

# CNN options
ENSEMBLE_SIZE = 10	# no. CNNs in ensemble
REGULARIZATION = 'anc'		# type of regularisation to use - anc (anchoring) reg (regularised) free (unconstrained)
BATCH_SIZE = 1


##------------------------------------------------------------------------------------
## Load loss and loss coefficients
from losses.combined_loss_ssim_grad_hist import combined_loss

COEF_SSIM = 1  # Strength of SSIM term
COEF_GRAD = 4  # Strength of gradient term
COEF_HIST = 0.2  # Strength of histogram term

loss = combined_loss(COEF_SSIM, COEF_GRAD, COEF_HIST)

##------------------------------------------------------------------------------------
## Patch location and data loader
from loaders.euv_aia_eit_loader import imageLoader

PATCH_PATH = '/d0/patches/val/'
TRAIN_PATH = '/d1/patches/trn/'
VAL_PATH = '/d0/patches/val/'

L = 1536*(196 + 169)//BATCH_SIZE 
L1 = 451*(196 + 169)//BATCH_SIZE

# data options
N_DATA = 4*(L + L1)*BATCH_SIZE 	# no. training + val data points


##------------------------------------------------------------------------------------
## Optimizer
adam = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5)


if __name__ == "__main__":

	patch_num = []
	for root,dirs,files in os.walk(PATCH_PATH):
		for file in files:
				if file.startswith('eit') and file[4:12]==f[3:11]:
					patches_e = pickle.load(open(root+file, "rb" ))
					if np.sum(patches_e[:,:,1]<1)>=0:#.2*64*64:
						patch_num.append(file[13:18])

	# create the NNs
	CNNs=[]
	for m in range(ENSEMBLE_SIZE):
		CNNs.append(fn_make_CNN(reg=REGULARIZATION))
			#sgd = SGD(lr=0.0001, momentum=0.9, nesterov=True)
		CNNs[m].compile(optimizer=adam, loss = loss, metrics=[loss],run_eagerly=True)

	print(CNNs[-1].summary())


	for m in range(ENSEMBLE_SIZE):
		print('-- training: ' + str(m+1) + ' of ' + str(ENSEMBLE_SIZE) + ' CNNs --') 
		checkpoint = ModelCheckpoint("/d0/models/eit_aia_sr_big_abae"+str(m+1).zfill(2)+".h5", monitor='val_combined_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
		#early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')#, baseline=None, restore_best_weights=True
		history = CNNs[m].fit(imageLoader(TRAIN_PATH, BATCH_SIZE, patch_num, FD_PATH_TRN, rng), batch_size = 4*BATCH_SIZE, steps_per_epoch = L, epochs = 10,callbacks=[checkpoint], validatioN_DATA=imageLoader(VAL_PATH, BATCH_SIZE, patch_num, FD_PATH_VAL, rng), validation_steps = L1,initial_epoch=epoch-1)
