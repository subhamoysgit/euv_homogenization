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
OUTPUT_FILES = ['eit_aia_sr_abae_small_LAMBDA_01_VAR_1_','eit_aia_sr_abae_small_LAMBDA_01_VAR_d1_','eit_aia_sr_abae_small_LAMBDA_0001_VAR_d1_']
PARAMS = ['(V = 1, L = 0.01)','(V = 0.1, L = 0.01)','(V = 0.1, L = 0.0001)']
COLORS = [['k','r'],['g','b'],['#E69F00','#56B4E9']]
ENSEMBLE_SIZE = 4

if __name__ == "__main__":
	loss = np.zeros((len(OUTPUT_FILES),ENSEMBLE_SIZE,10))
	val_loss = np.zeros((len(OUTPUT_FILES),ENSEMBLE_SIZE,10))
	mse = np.zeros((len(OUTPUT_FILES),ENSEMBLE_SIZE,10))
	val_mse = np.zeros((len(OUTPUT_FILES),ENSEMBLE_SIZE,10))
	for n in range(len(OUTPUT_FILES)):
		for m in range(ENSEMBLE_SIZE):
			t = pickle.load(open(OUTPUT_FOLDER + OUTPUT_FILES[n] + str(m+1).zfill(2) +'_'+'loss.p','rb'))
			loss[n,m,:] = t
			v = pickle.load(open(OUTPUT_FOLDER + OUTPUT_FILES[n] + str(m+1).zfill(2) +'_'+'val_loss.p','rb'))
			val_loss[n,m,:] = v
			tm = pickle.load(open(OUTPUT_FOLDER + OUTPUT_FILES[n] + str(m+1).zfill(2) +'_'+'mse.p','rb'))
			mse[n,m,:] = tm
			vm = pickle.load(open(OUTPUT_FOLDER + OUTPUT_FILES[n] + str(m+1).zfill(2) +'_'+'val_mse.p','rb'))
			val_mse[n,m,:] = vm

	epochs = np.linspace(1,10,10)
	plt.subplot(1,2,1)
	for n in range(len(OUTPUT_FILES)):
		plt.fill_between(epochs,loss[n,:,:].min(axis=0),loss[n,:,:].max(axis=0),color=COLORS[n][0],alpha=0.5,label='train '+PARAMS[n])
		plt.fill_between(epochs,val_loss[n,:,:].min(axis=0),val_loss[n,:,:].max(axis=0),color=COLORS[n][1],alpha=0.5,label='val '+PARAMS[n])
	plt.ylabel('total loss')
	plt.xlabel('epochs')
	plt.ylim([val_mse.min(),loss.max()])
	#plt.text(4,10**12,'V = 1, L = 0.01')
	plt.yscale('log')
	plt.legend(frameon = False)
	plt.subplot(1,2,2)
	for n in range(len(OUTPUT_FILES)):
		plt.fill_between(epochs,mse[n,:,:].min(axis=0),mse[n,:,:].max(axis=0),color=COLORS[n][0],alpha=0.5,label='train '+PARAMS[n])
		plt.fill_between(epochs,val_mse[n,:,:].min(axis=0),val_mse[n,:,:].max(axis=0),color=COLORS[n][1],alpha=0.5,label='val '+PARAMS[n])
	plt.ylabel('mse')
	plt.xlabel('epochs')
	plt.ylim([val_mse.min(),loss.max()])
	#plt.text(4,10**12,'V = 1, L = 0.01')
	plt.yscale('log')
	plt.legend(frameon = False)
	plt.show()
