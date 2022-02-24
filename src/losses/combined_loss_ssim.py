import os
import sys
import tensorflow as tf

# Append source folder to system path
_MODEL_DIR = os.path.abspath(__file__).split('/')[:-1]
_SRC_DIR = os.path.join('/',*_MODEL_DIR[:-1])
sys.path.append(_SRC_DIR)

## Native Modules
from losses import term_mse


def combined_loss(coef_ssim):
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

		return term_mse.mse_loss(y_true, y_pred) + coef_ssim*tf.image.ssim(y_true, y_pred, max_val=10.0)

	return loss