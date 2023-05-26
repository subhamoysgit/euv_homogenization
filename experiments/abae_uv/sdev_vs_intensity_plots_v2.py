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
import pickle as p
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
VAL_PATH = '/d1/patches/trn/'  # Validation data path


# Augmentation
VFLIP = True  # Vertical flip
HFLIP = True  # Horizontal flip

##------------------------------------------------------------------------------------
## Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5)

ANCHOR_PARAMS  = '_GLOROT_UNIF_last_LAMBDA_SCALED_01_ST_1_'
ANCHOR_PARAMS_1  = '_GLOROT_UNIF_last_LAMBDA_SCALED_001_ST_10_'
FOLDER = '/d0/models/'
dict = {'TRN':'TRAINING SET','VAL':'VALIDATION SET','TST':'TEST SET'}
dict_prof = {}
if __name__ == "__main__":
	k = 1
	for i in ['TRN','VAL','TST']:
		euv20 = p.load(open(FOLDER + "EUV_"+i+"_SDEV_70k_"+ANCHOR_PARAMS+"1000","rb"))
		euv20_1 = p.load(open(FOLDER + "EUV_"+i+"_SDEV_70k_"+ANCHOR_PARAMS_1+"1000","rb"))
		euv70 = p.load(open(FOLDER + "EUV_"+i+"_SDEV_70k_1000","rb"))
		H20, xedges, yedges = np.histogram2d(euv20[:,0],euv20[:,1], bins=(2000, 2000),range = [[0.0,1.0],[0,0.05]],density=True)
		H20_1, xedges, yedges = np.histogram2d(euv20_1[:,0],euv20_1[:,1], bins=(2000, 2000),range = [[0.0,1.0],[0,0.05]],density=True)
		H70, xedges, yedges = np.histogram2d(euv70[:,0],euv70[:,1], bins=(2000, 2000),range = [[0.0,1.0],[0,0.05]],density=True)
		mn = 0*np.min(H70)
		mx = 0.7*np.max(H70)
		dict_prof[i,0] = np.sum(H20,axis = 0)
		dict_prof[i,1] = np.sum(H20_1,axis = 0)
		dict_prof[i,2] = np.sum(H70,axis = 0)
		plt.subplot(3,3,k)
		plt.hist2d(euv20[:,0],euv20[:,1],range = [[0.0,1.0],[0,0.05]], bins=(2000,2000),density=True,vmin=mn,vmax=mx)
		if k==1:
			plt.ylabel('ensemble SD')
		plt.xlim([0.0,1.0])
		plt.ylim([0,0.05])
		plt.xticks([])
		plt.title(dict[i]+',70k,SD=1,LAMBDA=0.01')
		plt.subplot(3,3,k+3)
		plt.hist2d(euv20_1[:,0],euv20_1[:,1],range = [[0.0,1.0],[0,0.05]],bins=(2000,2000),density=True,vmin=mn,vmax=mx)
		if k==1:
			plt.ylabel('ensemble SD')
		plt.xlim([0.0,1.0])
		plt.ylim([0,0.05])
		plt.xticks([])
		plt.title(dict[i]+',70k,SD=10,LAMBDA=0.001')
		plt.subplot(3,3,k+6)
		plt.hist2d(euv70[:,0],euv70[:,1],range = [[0.0,1.0],[0,0.05]],bins=(2000,2000),density=True,vmin=mn,vmax=mx)
		if k==1:
			plt.ylabel('ensemble SD')
		plt.xlim([0.0,1.0])
		plt.ylim([0,0.05])
		plt.xlabel('input intensity')
		plt.title(dict[i]+',70k,SD=1,LAMBDA=0.001')
		k = k+1
	plt.figure(figsize=(15,5))
	f = ['TRN','VAL','TST']
	xx = np.linspace(0,0.05,2000)
	labels = ['SD = 1,LAMBDA = 0.01','SD = 10,LAMBDA = 0.001','SD = 1,LAMBDA = 0.001']
	ANC_PARAMS = [ANCHOR_PARAMS,ANCHOR_PARAMS_1,'']
	for n in range(3):
		euv1 = p.load(open(FOLDER + "EUV_TRN_SDEV_70k_"+ANC_PARAMS[n]+"1000","rb"))
		euv2 = p.load(open(FOLDER + "EUV_VAL_SDEV_70k_"+ANC_PARAMS[n]+"1000","rb"))
		euv3 = p.load(open(FOLDER + "EUV_TST_SDEV_70k_"+ANC_PARAMS[n]+"1000","rb"))
		plt.subplot(1,3,n+1)
		# plt.plot(xx,dict_prof[f[0],n],label='Training')
		# plt.plot(xx,dict_prof[f[1],n],label='Validation')
		# plt.plot(xx,dict_prof[f[2],n],label='Test')
		plt.hist(euv1[:,1],bins=2000,range=[0,0.05],density=True,histtype='step',label='Training')
		plt.hist(euv2[:,1],bins=2000,range=[0,0.05],density=True,histtype='step',label='Validation')
		plt.hist(euv3[:,1],bins=2000,range=[0,0.05],density=True,histtype='step',label='Test')
		plt.xlabel('sd')
		plt.title(labels[n])
		plt.yscale('log')
		plt.legend(frameon = False)
	plt.show()

	plt.figure(figsize=(15,15))
	f = ['TRN','VAL','TST']
	xx = np.linspace(0,0.05,2000)
	labels = ['SD = 1,LAMBDA = 0.01','SD = 10,LAMBDA = 0.001','SD = 1,LAMBDA = 0.001']
	ANC_PARAMS = [ANCHOR_PARAMS,ANCHOR_PARAMS_1,'']
	for n in range(3):
		euv1 = p.load(open(FOLDER + "EUV_TRN_SDEV_70k_"+ANC_PARAMS[n]+"1000","rb"))
		euv2 = p.load(open(FOLDER + "EUV_VAL_SDEV_70k_"+ANC_PARAMS[n]+"1000","rb"))
		euv3 = p.load(open(FOLDER + "EUV_TST_SDEV_70k_"+ANC_PARAMS[n]+"1000","rb"))
		ind11 = np.where(np.abs(euv1[:,0]-0.64)<0.01)[0]
		print(ind11)
		ind12 = np.where(np.abs(euv1[:,0]-0.7)<0.01)[0]
		ind13 = np.where(np.abs(euv1[:,0]-0.76)<0.01)[0]
		ind21 = np.where(np.abs(euv2[:,0]-0.64)<0.01)[0]
		ind22 = np.where(np.abs(euv2[:,0]-0.7)<0.01)[0]
		ind23 = np.where(np.abs(euv2[:,0]-0.76)<0.01)[0]
		ind31 = np.where(np.abs(euv3[:,0]-0.64)<0.01)[0]
		ind32 = np.where(np.abs(euv3[:,0]-0.7)<0.01)[0]
		ind33 = np.where(np.abs(euv3[:,0]-0.76)<0.01)[0]
		plt.subplot(3,3,3*n+1)
		# plt.plot(xx,dict_prof[f[0],n],label='Training')
		# plt.plot(xx,dict_prof[f[1],n],label='Validation')
		# plt.plot(xx,dict_prof[f[2],n],label='Test')
		plt.hist(euv1[ind11,1],bins=200,range=[0,0.05],density=True,histtype='step',label='Training')
		plt.hist(euv2[ind21,1],bins=200,range=[0,0.05],density=True,histtype='step',label='Validation')
		plt.hist(euv3[ind31,1],bins=200,range=[0,0.05],density=True,histtype='step',label='Test')
		if n<=1: plt.xticks([])
		if n==2: plt.xlabel('sd')
		plt.title(labels[n]+', I = '+str(0.64))
		plt.yscale('log')
		plt.legend(frameon = False)
		plt.subplot(3,3,3*n+2)
		plt.hist(euv1[ind12,1],bins=200,range=[0,0.05],density=True,histtype='step',label='Training')
		plt.hist(euv2[ind22,1],bins=200,range=[0,0.05],density=True,histtype='step',label='Validation')
		plt.hist(euv3[ind32,1],bins=200,range=[0,0.05],density=True,histtype='step',label='Test')
		if n<=1: plt.xticks([])
		if n==2: plt.xlabel('sd')
		plt.title(labels[n]+', I = '+str(0.7))
		plt.yscale('log')
		plt.legend(frameon = False)
		plt.subplot(3,3,3*n+3)
		plt.hist(euv1[ind13,1],bins=200,range=[0,0.05],density=True,histtype='step',label='Training')
		plt.hist(euv2[ind23,1],bins=200,range=[0,0.05],density=True,histtype='step',label='Validation')
		plt.hist(euv3[ind33,1],bins=200,range=[0,0.05],density=True,histtype='step',label='Test')
		if n<=1: plt.xticks([])
		if n==2: plt.xlabel('sd')
		plt.title(labels[n]+', I = '+str(0.76))
		plt.yscale('log')
		plt.legend(frameon = False)
	plt.show()