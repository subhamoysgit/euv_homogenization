import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import *

def differentiable_histogram(x, dl = 0.1, normalization = 1000, noise_level = 20, lim = 3000):

	"""Differentiable 

	Parameters
	----------
	y_true : tf.tensor
		Target image data
	y_pred : tf.tensor
		Predicted image data
	"""

	x = x*normalization
	patch_size = x.shape[2]
	# dl = 0.1
	# lim = 3000 
	# normalization = 1 
	# noise_level = 20
	lim = np.log10(lim)  # Maximum value

	bins = np.round(np.power(10, np.arange(1, lim + dl, dl)), 2)
	bins = bins - 10 + noise_level
	
	# Defining centers and bin widths for the learnable histogram
	centers = (bins[1:] + bins[0:-1]) / 2
	widths = (bins[1:] - bins[0:-1])
	nonzero = np.ones(centers.shape[0])
	weight1 = nonzero[None, None, None,:]
	bias1 = -centers * nonzero
	diag = -nonzero / widths
	weight2 = np.expand_dims(np.expand_dims(np.diag(diag), axis=0), axis=0)
	bias2 = nonzero
	conv = Conv2D(centers.shape[0], 1, weights= [weight1, bias1])(x)
	conv = K.abs(conv)
	conv = Conv2D(centers.shape[0], 1, weights= [weight2, bias2])(conv)
	conv = Activation('relu')(conv)
	hist = AveragePooling2D(pool_size=(patch_size, patch_size))(conv)
	
	return hist


def histogram_loss(y_true,y_pred):
	"""Calculate the Gradient magnitude loss

	Parameters
	----------
	y_true : tf.tensor
		Target image data
	y_pred : tf.tensor
		Predicted image data
	"""

	x = x*1000
	patch_size = x.shape[2]
	dl = 0.1
	lim = 3000 
	normalization = 1 
	noise_level = 20
	lim = np.log10(lim)  # Maximum value

	bins = np.round(np.power(10, np.arange(1, lim + dl, dl)), 2)
	bins = bins - 10 + noise_level
	# Defining centers and bin widths for the learnable histogram
	centers = (bins[1:] + bins[0:-1]) / 2
	widths = (bins[1:] - bins[0:-1])
	nonzero = np.ones(centers.shape[0])
	weight1 = nonzero[None, None, None,:]
	bias1 = -centers * nonzero
	diag = -nonzero / widths
	weight2 = np.expand_dims(np.expand_dims(np.diag(diag), axis=0), axis=0)
	bias2 = nonzero
	conv = Conv2D(centers.shape[0], 1, weights= [weight1, bias1])(x)
	conv = K.abs(conv)
	conv = Conv2D(centers.shape[0], 1, weights= [weight2, bias2])(conv)
	conv = Activation('relu')(conv)
	hist = AveragePooling2D(pool_size=(patch_size, patch_size))(conv)
	#print(patch_size)
	return hist
