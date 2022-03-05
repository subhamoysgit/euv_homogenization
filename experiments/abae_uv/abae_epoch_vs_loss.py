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
import pickle
import numpy as np
import matplotlib.pyplot as plt
OUTPUT_FOLDER = '/d0/models/'
OUTPUT_FILE = 'eit_aia_sr_abae_small_LAMBDA_01_VAR_1_'
ENSEMBLE_SIZE = 7

if __name__ == "__main__":
	loss = np.zeros((7,10))
	val_loss = np.zeros((7,10))
	mse = np.zeros((7,10))
	val_mse = np.zeros((7,10))
	for m in range(ENSEMBLE_SIZE):
		t = pickle.load(open(OUTPUT_FOLDER + OUTPUT_FILE + str(m+1).zfill(2) +'_'+'loss.p','rb'))
		loss[m,:] = t
		v = pickle.load(open(OUTPUT_FOLDER + OUTPUT_FILE + str(m+1).zfill(2) +'_'+'val_loss.p','rb'))
		val_loss[m,:] = v
		tm = pickle.load(open(OUTPUT_FOLDER + OUTPUT_FILE + str(m+1).zfill(2) +'_'+'mse.p','rb'))
		mse[m,:] = tm
		vm = pickle.load(open(OUTPUT_FOLDER + OUTPUT_FILE + str(m+1).zfill(2) +'_'+'val_mse.p','rb'))
		val_mse[m,:] = vm
	epochs = np.linspace(1,10,10)
	plt.subplot(1,2,1)
	plt.fill_between(epochs,loss.min(axis=0),loss.max(axis=0),color='k',alpha=0.5,label='train')
	plt.fill_between(epochs,val_loss.min(axis=0),val_loss.max(axis=0),color='r',alpha=0.5,label='val')
	plt.ylabel('total loss')
	plt.xlabel('epochs')
	plt.yscale('log')
	plt.legend(frameon = False)
	plt.subplot(1,2,2)
	plt.fill_between(epochs,mse.min(axis=0),mse.max(axis=0),color='k',alpha=0.5,label='train')
	plt.fill_between(epochs,val_mse.min(axis=0),val_mse.max(axis=0),color='r',alpha=0.5,label='val')
	plt.ylabel('mse')
	plt.xlabel('epochs')
	plt.yscale('log')
	plt.legend(frameon = False)
	plt.show()