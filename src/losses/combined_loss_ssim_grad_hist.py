import os

_MODEL_DIR = os.path.abspath(__file__).split('/')[:-1]
_SRC_DIR = os.path.join('/',*_MODEL_DIR[:-1])

import tensorflow as tf

## Native
from losses import term_gradient, term_histogram

def combined_loss(y_true,y_pred):
	return tf.math.reduce_mean(tf.square(y_true-y_pred)) - 0.001*tf.image.ssim(y_true, y_pred, max_val=10.0)


def combined_loss(coef_ssim, coef_grad, coef_hist, dl = 0.1, normalization = 1000.0, noise_level = 20.0, lim = 3000.0):
	"""parent function setting the weigths used in the combined ssim+grad+hist+mse loss

	Parameters
	----------
	coef_ssim : float
		Weight of SSIM loss
	coef_grad : float
		Weight of gradient loss
	coef_hist : float
		Weight of histogram loss
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
		Combined loss funcion
	"""

	def loss(y_true, y_pred):

		return (tf.math.reduce_mean(tf.square(y_true-y_pred))
				  + coef_ssim*tf.image.ssim(y_true, y_pred, max_val=10.0)
				  + coef_grad*term_gradient.gradient_loss(y_true-y_pred)
				  + coef_hist*term_histogram.histogram_loss(y_true,y_pred)
				 )

	return loss