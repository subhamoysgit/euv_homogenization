import tensorflow as tf

def gradient_loss(y_true,y_pred):
	"""Calculate the Gradient magnitude loss

	Parameters
	----------
	y_true : tf.tensor
		Target image data
	y_pred : tf.tensor
		Predicted image data
	"""

	dx_true, dy_true = tf.image.image_gradients(y_true)
	dx_pred, dy_pred = tf.image.image_gradients(y_pred)
	return tf.math.reduce_mean(tf.square(dx_true - dx_pred)) + tf.math.reduce_mean(tf.square(dy_true - dy_pred))
