import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import *

def differentiable_histogram(x, dl = 0.1, normalization = 1000.0, noise_level = 20.0, lim = 3000.0):

	"""Differentiable histogram to use for images with wide dynamic ranges.  Bins are uniformly spaced
	   in logarithmic space, and non uniform in linear space.  Histogram is built using convolution layers and 
	   triangular kernels.  More info:  https://arxiv.org/abs/1804.09398

	Parameters
	----------
	x : tf.tensor
		Input data	
	dl : float
		Width of each logaritmic bin/
	normalization : float
		Normalization function to be applied to the data, this affects the relationship between the data and the bins.
	noise_level : float
		Value below which data values are ignored and bin counts set to zero.
	lim : float
		Upper histogram limit beyond which data are not taken into consideration.

	Returns
	-------
	hist : tf.tensor
		autograd vector with histogram		
	"""

	x = x*normalization
	patch_size = x.shape[2]
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
	"""Calculate the histogram loss

	Parameters
	----------
	y_true : tf.tensor
		Target image data
	y_pred : tf.tensor
		Predicted image data

	Returns
	-------
		histogram loss term
	"""

	return tf.math.reduce_mean(tf.square(differentiable_histogram(y_true)-differentiable_histogram(y_pred)))
