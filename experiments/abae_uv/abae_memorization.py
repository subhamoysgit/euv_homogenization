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
BATCH_SIZE = 10  # Batch Size
EPOCH0 = 1  # First epoch

DATA_NOISE = 0.1 # noise variance as mean of aia patch hist
W_VAR_I = 0.1 # variance of the anchor weights
W_LAMBDA_I = 0.01 # Strength of the regularization term for anchor weights
B_VAR_I = W_VAR_I # variance of the anchor biases 
B_LAMBDA_I = 0.01 # Strength of the regularization term for anchor biases


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
OUTPUT_FILE = 'eit_aia_sr_abae_small_LAMBDA_01_VAR_d1_'
TRAIN_DATE_RANGE = [20140101,20140430]
VAL_DATE_RANGE = [20160101,20160131]

if __name__ == "__main__":

	nTrain, nVal = imageIndexer(TRAIN_PATH, VAL_PATH, trainDateRange = TRAIN_DATE_RANGE, valDateRange = VAL_DATE_RANGE)
	print(nTrain)
	print(nVal)
	# create the NNs
	CNNs=[]
	for m in range(ENSEMBLE_SIZE):
		CNNs.append(make_CNN(reg=REGULARIZATION, features=32, rng=rng, W_var_i=W_VAR_I, W_lambda_i=W_LAMBDA_I, b_var_i=B_VAR_I, b_lambda_i=B_LAMBDA_I))
		CNNs[m].compile(optimizer=optimizer, loss = 'mse', metrics=['mse'], run_eagerly=True)

	print(CNNs[-1].summary())

	for m in range(ENSEMBLE_SIZE):
		print('-- training: ' + str(m+1) + ' of ' + str(ENSEMBLE_SIZE) + ' CNNs --') 
		checkpoint = ModelCheckpoint(OUTPUT_FOLDER + OUTPUT_FILE + str(m+1).zfill(2) +'_'+'{epoch:02d}.h5', monitor='val_loss', verbose=1, save_weights_only=True, mode='auto', save_freq='epoch')
		history = CNNs[m].fit(imageLoader(TRAIN_PATH, BATCH_SIZE, DateRange = TRAIN_DATE_RANGE, rng=rng, vflip=VFLIP, hflip=HFLIP), batch_size = ((VFLIP+HFLIP)**2)*BATCH_SIZE, steps_per_epoch = nTrain//BATCH_SIZE, epochs = 10, callbacks=[checkpoint], validation_data=imageLoader(VAL_PATH, BATCH_SIZE, DateRange = VAL_DATE_RANGE, rng=rng, vflip=VFLIP, hflip=HFLIP), validation_steps=nVal//BATCH_SIZE, initial_epoch=EPOCH0-1)
		pickle.dump(history.history['loss'],open(OUTPUT_FOLDER + OUTPUT_FILE + str(m+1).zfill(2) +'_'+'loss.p','wb'))
		pickle.dump(history.history['val_loss'],open(OUTPUT_FOLDER + OUTPUT_FILE + str(m+1).zfill(2) +'_'+'val_loss.p','wb'))
		pickle.dump(history.history['mse'],open(OUTPUT_FOLDER + OUTPUT_FILE + str(m+1).zfill(2) +'_'+'mse.p','wb'))
		pickle.dump(history.history['val_mse'],open(OUTPUT_FOLDER + OUTPUT_FILE + str(m+1).zfill(2) +'_'+'val_mse.p','wb'))
