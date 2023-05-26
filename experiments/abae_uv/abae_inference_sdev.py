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
VAL_PATH = '/d1/patches/trn/'  # Validation data path


# Augmentation
VFLIP = True  # Vertical flip
HFLIP = True  # Horizontal flip

##------------------------------------------------------------------------------------
## Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5)


OUTPUT_FOLDER = '/d0/models/'
OUTPUT_FILE = 'eit_aia_sr_abae_small_GLOROT_UNIF_last_LAMBDA_SCALED_001_ST_1_'#'eit_aia_sr_big_v17'#'eit_aia_sr_abae_small_LAMBDA_01_VAR_1_'
TRAIN_DATE_RANGE = [20140101,20141231]
VAL_DATE_RANGE = [20170101,20170229]

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

from sunpy.map import Map
file_m = '/d1/sep_dataset/videos/positives/mag_200612142107.p'
mag = pickle.load(open(file_m,'rb'))
mag = mag[:,:,0]/65535
mag_patch = mag[100:164,100:164]

print(np.min(mag_patch))
print(np.max(mag_patch))
if __name__ == "__main__":
	PATCH_NAME = EIT_VAL[100]
	print(PATCH_NAME)
	eit = pickle.load(open(PATCH_NAME, "rb" ))
	aia = pickle.load(open(PATCH_NAME[:16]+'aia'+PATCH_NAME[19:], "rb" ))
	print(np.min(eit[:,:,0]))
	print(np.max(eit[:,:,0]))
	X = np.zeros((1,64,64,2))
	Y = np.zeros((1,256,256,1))
	prof = eit[:,:,1]
	X[0,:,:,0] = eit[:,:,0]#np.min(eit[:,:,0]) + (np.max(eit[:,:,0])-np.min(eit[:,:,0]))*mag_patch #eit[:,:,0]
	X[0,:,:,1] = prof[:,:]
	Y[0,:,:,0] = aia[:,:]
	nTrain, nVal = imageIndexer(TRAIN_PATH, VAL_PATH, trainDateRange = TRAIN_DATE_RANGE, valDateRange = VAL_DATE_RANGE)
	print(nTrain)
	print(nVal)
	best_epochs = np.zeros(ENSEMBLE_SIZE)
	for m in range(ENSEMBLE_SIZE):
		p = pickle.load(open(OUTPUT_FOLDER + OUTPUT_FILE + str(m+1).zfill(2) +'_'+'val_mse.p','rb'))
		best_epochs[m] = 1 + np.argmin(p)
	print(best_epochs)
	# create the NNs
	CNNs=[]
	out = np.zeros((ENSEMBLE_SIZE,256,256))
	for m in range(ENSEMBLE_SIZE):
		CNNs.append(make_CNN(reg=REGULARIZATION, features=32, rng=rng, W_var_i=W_VAR_I, W_lambda_i=W_LAMBDA_I, b_var_i=B_VAR_I, b_lambda_i=B_LAMBDA_I))
		CNNs[m].load_weights(OUTPUT_FOLDER + OUTPUT_FILE + str(m+1).zfill(2) +'_'+str(int(best_epochs[m])).zfill(2)+'.h5')
		pred = CNNs[m].predict(X)
		out[m,:,:] = pred[0,:,:,0]
		
	input_intensity = []
	output_sdev = []
	for i in range(64):
		for j in range(64):
			o = np.std(out[:,4*i:4*(i+1),4*j:4*(j+1)])
			input_intensity.append(X[0,i,j,0])
			output_sdev.append(o)
	#plt.plot(input_intensity,output_sdev,'.k')
	H, xedges, yedges = np.histogram2d(input_intensity,output_sdev, bins=(20, 20),range = [[np.min(input_intensity), np.max(input_intensity)], [np.min(output_sdev), np.max(output_sdev)]])
	print([[np.min(input_intensity), np.max(input_intensity)], [np.min(output_sdev), np.max(output_sdev)]])
	pickle.dump(H.T,open('/d0/models/euv_test_std_21k.p','wb'))
	#plt.hist2d(input_intensity,output_sdev,bins=(20,20))
	plt.imshow(H.T)
	plt.xlabel('input intensity')
	plt.ylabel('ensemble stdev over 4x4 blocks')
	plt.title ('EUV')
	plt.show()
	# fig = plt.figure(figsize=(40,40))
	# ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.4])
	# ax2 = fig.add_axes([0.1, 0.5, 0.4, 0.4])
	# ax3 = fig.add_axes([0.5, 0.1, 0.4, 0.4],sharey = ax1)
	# ax4 = fig.add_axes([0.5, 0.5, 0.4, 0.4],sharey = ax2)
	# mag = pickle.load(open('/d0/models/mag_std.p','rb'))
	# euv = pickle.load(open('/d0/models/euv_std.p','rb'))
	# mag_b = pickle.load(open('/d0/models/mag_std_71k.p','rb'))
	# euv_b = pickle.load(open('/d0/models/euv_std_71k.p','rb'))
	# ax1.imshow(np.flip(mag,0),extent=[0.725, 0.925,0.0050, 0.0275],aspect = 0.2/0.0225)
	# ax1.set_yticks([0.01,0.015,0.02])
	# ax1.set_ylabel('ensemble stdev over 4x4 blocks')
	# ax1.set_xlabel('input intensity')
	# ax1.text(0.85,0.007,'Magnetogram',color = 'yellow')
	# ax2.imshow(np.flip(euv,0),extent=[0.725, 0.925,0.0050, 0.0275],aspect = 0.2/0.0225)
	# ax2.set_yticks([0.01,0.015,0.02])
	# ax2.set_xticks([])
	# ax2.set_ylabel('ensemble stdev over 4x4 blocks')
	# ax2.text(0.85,0.007,'EUV',color = 'yellow')
	# ax2.set_title('20K training patches')
	# ax3.imshow(np.flip(mag_b,0),extent=[0.725, 0.925,0.0050, 0.0275],aspect = 0.2/0.0225)
	# ax3.set_yticks([])
	# ax3.set_xlabel('input intensity')
	# ax3.text(0.85,0.007,'Magnetogram',color = 'yellow')
	# ax4.imshow(np.flip(euv_b,0),extent=[0.725, 0.925,0.0050, 0.0275],aspect = 0.2/0.0225)
	# ax4.set_yticks([0.01,0.015,0.02])
	# ax4.set_xticks([])
	# ax4.set_yticks([])
	# ax4.text(0.85,0.007,'EUV',color = 'yellow')
	# ax4.set_title('70K training patches')
	# plt.show()
	# plt.figure()
	# ind1 = np.argmax(mag[5,:])
	# ind2 = np.argmax(mag_b[5,:])
	# print([ind1,ind2])
	# plt.plot(0.0050 + np.linspace(0,19,20)*0.0225/19,euv[:,ind1]/np.sum(euv[:,ind1]),'--k',label='euv 20')
	# plt.plot(0.0050 + np.linspace(0,19,20)*0.0225/19,mag[:,ind1]/np.sum(mag[:,ind1]),'--r',label='mag 20')
	# plt.plot(0.0050 + np.linspace(0,19,20)*0.0225/19,euv_b[:,ind2]/np.sum(euv_b[:,ind2]),'-k',label='euv 70')
	# plt.plot(0.0050 + np.linspace(0,19,20)*0.0225/19,mag_b[:,ind2]/np.sum(mag_b[:,ind2]),'-r',label='mag 70')
	# plt.xlabel('ensemble stdev over 4x4 blocks')
	# plt.legend(frameon = False)
	# plt.show()