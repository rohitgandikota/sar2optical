from keras import backend as K

def feature_reconstruction_loss(base, output):
	"""
	Feature reconstruction loss function. Encourages the 
	output img to be perceptually similar to the base image.
	"""
	return K.sum(K.square(output - base))

def gram_matrix(x):
	"""
	Computes the outer-product of the input tensor x.

	Input
	-----
	- x: input tensor of shape (C x H x W)

	Returns
	-------
	- x . x^T

	Note that this can be computed efficiently if x is reshaped
	as a tensor of shape (C x H*W).
	"""
	# assert K.ndim(x) == 3
	if K.image_data_format() == 'channels_first':
		features = K.batch_flatten(x)
	else:
		features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
	return K.dot(features, K.transpose(features))

def style_reconstruction_loss(base, output, img_nrows, img_ncols):
	"""
	Style reconstruction loss. Encourage the output img 
	to have same stylistic features as style image. Does not
	preserve spatial structure however.
	"""
	H, W, C = img_nrows, img_ncols, 3
	gram_base = gram_matrix(base)
	gram_output = gram_matrix(output)
	factor = 1.0 / float((2*C*H*W)**2)
	out = factor * K.sum(K.square(gram_output - gram_base))
	return out

def total_variation_loss(x, img_nrows, img_ncols):
	"""
	Total variational loss. Encourages spatial smoothness 
	in the output image.
	"""
	H, W = img_nrows, img_ncols
	if  K.image_data_format()  == 'channels_first':
		a = K.square(x[:, :, :H-1, :W-1] - x[:, :, 1:, :W-1])
		b = K.square(x[:, :, :H-1, :W-1] - x[:, :, :H-1, 1:])
	else:	
		a = K.square(x[:, :H-1, :W-1, :] - x[:, 1:, :W-1, :])
		b = K.square(x[:, :H-1, :W-1, :] - x[:, :H-1, 1:, :])

	return K.sum(K.pow(a + b, 1.25))
