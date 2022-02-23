import tensorflow as tf

def mse_loss(y_true,,y_pred):
	"""Calculate the mse loss

	Parameters
	----------
	y_true : tf.tensor
		Target image data
	y_pred : tf.tensor
		Predicted image data

	Returns
	-------
		mse loss term
	"""

	return tf.math.reduce_mean(tf.square(y_true-y_pred))
