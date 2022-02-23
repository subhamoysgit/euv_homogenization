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
import tensorflow as tf
from keras.callbacks import ModelCheckpoint


##------------------------------------------------------------------------------------
## Random seed initialization
SEED_VALUE = 42
rng = np.random.default_rng(SEED_VALUE)


##------------------------------------------------------------------------------------
## Load CNN model and CNN coefficients
from models.model_HighResnet_ABAE import make_CNN

# CNN options
ENSEMBLE_SIZE = 10  # no. CNNs in ensemble
REGULARIZATION = 'anc'  # type of regularisation to use - anc (anchoring) reg (regularised) free (unconstrained)
BATCH_SIZE = 1  # Batch Size
EPOCH0 = 1  # First epoch

W_VAR_I = 1.0 # Standard deviation of the anchor weights
W_LAMBDA_I = 0.5 # Strength of the regularization term for anchor weights
B_VAR_I = 1.0 # Standard deviation of the anchor biases 
B_LAMBDA_I = 0.5 # Strength of the regularization term for anchor biases


##------------------------------------------------------------------------------------
## Load loss and loss coefficients
from losses.combined_loss_ssim_grad_hist import combined_loss

COEF_SSIM = 1  # Strength of SSIM term
COEF_GRAD = 4  # Strength of gradient term
COEF_HIST = 0.2  # Strength of histogram term

loss = combined_loss(COEF_SSIM, COEF_GRAD, COEF_HIST)

##------------------------------------------------------------------------------------
## Patch location, data loader, and augmentation
from data_loaders.eit_aia_loader import imageIndexer, imageLoader

PATCH_PATH = '/d0/patches/val/'  # Patch location
TRAIN_PATH = '/d1/patches/trn/'  # Training data path
VAL_PATH = '/d0/patches/val/'  # Validation data path

# Augmentation
VFLIP = True  # Vertical flip
HFLIP = True  # Horizontal flip

##------------------------------------------------------------------------------------
## Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5)


if __name__ == "__main__":

	patch_num, nTrain, nVal = imageIndexer(PATCH_PATH, TRAIN_PATH, VAL_PATH)

	# create the NNs
	CNNs=[]
	for m in range(ENSEMBLE_SIZE):
		CNNs.append(make_CNN(reg=REGULARIZATION, features=32, rng=rng, W_var_i=W_VAR_I, W_lambda_i=W_LAMBDA_I, b_var_i=B_VAR_I, b_lambda_i=B_LAMBDA_I))
		CNNs[m].compile(optimizer=optimizer, loss = loss, metrics=[loss], run_eagerly=True)

	print(CNNs[-1].summary())

	for m in range(ENSEMBLE_SIZE):
		print('-- training: ' + str(m+1) + ' of ' + str(ENSEMBLE_SIZE) + ' CNNs --') 
		checkpoint = ModelCheckpoint("/d0/models/eit_aia_sr_big_abae"+str(m+1).zfill(2)+".h5", monitor='val_combined_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
		history = CNNs[m].fit(imageLoader(TRAIN_PATH, BATCH_SIZE, patch_num, rng=rng, vflip=VFLIP, hflip=HFLIP), batch_size = ((VFLIP+HFLIP)**2)*BATCH_SIZE, steps_per_epoch = nTrain//BATCH_SIZE, epochs = 10, callbacks=[checkpoint], validation_data=imageLoader(VAL_PATH, BATCH_SIZE, patch_num, rng=rng), validation_steps=nVal//BATCH_SIZE, initial_epoch=EPOCH0-1)
