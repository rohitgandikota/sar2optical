# load necesssary packages
from __future__ import print_function

import time
import numpy as np
from matplotlib.pyplot import imsave
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
from keras.applications import vgg16, vgg19
from keras.preprocessing.image import load_img
from utils import preprocess_image, deprocess_image
from losses import style_reconstruction_loss, feature_reconstruction_loss, total_variation_loss

class Neural_Styler(object):
	"""
	Mix the style of an image with the content of another.

	References
	----------
	- "A Neural Algorithm of Artistic Style", [arXiv:1508.06576]: Leon A. Gatys, 
	  Alexander S. Ecker and Matthias Bethge.
	"""
	def __init__(self, 
		     base_img_path,
		     style_img_path,
		     output_img_path,
		     output_width,
		     convnet, 
		     content_weight, 
		     style_weight, 
		     tv_weight,
		     content_layer,
		     style_layers,
		     iterations):
		"""
		Initialize and store parameters of the neural styler. Initialize the
		desired convnet and compute the 3 losses and gradients with respect to the
		output image.

		Params
		------
		- input_img: tensor containing: content_img, style_img and output_img.
		- convnet: [string], defines which VGG to use: vgg16 or vgg19.
		- style_layers: list containing name of layers to use for style
		  reconstruction. Defined in Gatys et. al but can be changed.
		- content_layer: string containing name of layer to use for content
		  reconstruction. Also defined in Gatys et. al.
		- content_weight: weight for the content loss.
		- style_weight: weight for the style loss.
		- tv_weight: weight for the total variation loss.
		- iterations: iterations for optimization algorithm
		- output_img_path: path to output image.

		Notes
		-----
		[1] If user specifies output width, then calculate the corresponding 
		    image height. Else, output image height and width should be the
		    same as that of the content image. Also note that style image
		    should be resized to whatever was decided above.

		[2] PIL returns (width, height) whereas numpy returns (height, width)
		    since nrows=height and ncols=width.

		[3] L_BFGS requires that loss and grad be two functions so we create
		    a keras function that computes the gradients and loss and return 
		    each separately using two different class methods.
		"""
		print('\nInitializing Neural Style model...')

		# store paths
		self.base_img_path = base_img_path
		self.style_img_path = style_img_path
		self.output_img_path = output_img_path

		# configuring image sizes [1, 2]
		print('\n\tResizing images...')
		self.width = output_width
		width, height = load_img(self.base_img_path).size
		new_dims = (height, width)

		# store shapes for future use
		self.img_nrows = height
		self.img_ncols = width

		if self.width is not None:
			# calculate new height
			num_rows = int(np.floor(float(height * self.width / width)))
			new_dims = (num_rows, self.width)

			# update the stored shapes
			self.img_nrows = num_rows
			self.img_ncols = self.width

		# resize content and style images to this desired shape
		self.content_img = K.variable(preprocess_image(self.base_img_path, new_dims))
		self.style_img = K.variable(preprocess_image(self.style_img_path, new_dims))

		# and also create output placeholder with desired shape
		if K.image_data_format() == 'channels_first':
			self.output_img = K.placeholder((1, 3, new_dims[0], new_dims[1]))
		else:
			self.output_img = K.placeholder((1, new_dims[0], new_dims[1], 3))

		# sanity check on dimensions
		print("\tSize of content image is: {}".format(K.int_shape(self.content_img)))
		print("\tSize of style image is: {}".format(K.int_shape(self.style_img)))
		print("\tSize of output image is: {}".format(K.int_shape(self.output_img)))

		# combine the 3 images into a single Keras tensor
		self.input_img = K.concatenate([self.content_img, 
						self.style_img, 
						self.output_img], axis=0)

		self.convnet = convnet
		self.iterations = iterations

		# store weights of the loss components
		self.content_weight = content_weight
		self.style_weight = style_weight
		self.tv_weight = tv_weight

		# store convnet layers
		self.content_layer = content_layer
		self.style_layers = style_layers

		# initialize the vgg16 model
		print('\tLoading {} model'.format(self.convnet.upper()))

		if self.convnet == 'vgg16':
			self.model = vgg16.VGG16(input_tensor=self.input_img, 
						 weights='imagenet', 
						 include_top=False)
		else:
			self.model = vgg19.VGG19(input_tensor=self.input_img, 
						 weights='imagenet', 
						 include_top=False)

		print('\tComputing losses...')
		# get the symbolic outputs of each "key" layer (we gave them unique names).
		outputs_dict = dict([(layer.name, layer.output) for layer in self.model.layers])

		# extract features only from the content layer
		content_features = outputs_dict[self.content_layer]

		# extract the activations of the base image and the output image
		base_image_features = content_features[0, :, :, :] 	# 0 corresponds to base
		combination_features = content_features[2, :, :, :]	# 2 coresponds to output

		# calculate the feature reconstruction loss
		content_loss = self.content_weight * \
					   feature_reconstruction_loss(base_image_features, 
								       combination_features)
		
		# for each style layer compute style loss
		# total style loss is then weighted sum of those losses
		temp_style_loss = K.variable(0.0)
		weight = 1.0 / float(len(self.style_layers))

		for layer in self.style_layers:
			# extract features of given layer
			style_features = outputs_dict[layer]
			# from those features, extract style and output activations
			style_image_features = style_features[1, :, :, :]
			output_style_features = style_features[2, :, :, :]
			temp_style_loss = temp_style_loss + (weight * \
					  style_reconstruction_loss(style_image_features, 
								    output_style_features,
								    self.img_nrows, 
								    self.img_ncols))
		style_loss = self.style_weight * temp_style_loss

		# compute total variational loss
		tv_loss = self.tv_weight * total_variation_loss(self.output_img, 
								self.img_nrows, 
								self.img_ncols)

		# composite loss
		total_loss = content_loss + style_loss + tv_loss

		# compute gradients of output img with respect to loss
		print('\tComputing gradients...')
		grads = K.gradients(total_loss, self.output_img)

		outputs = [total_loss]
		if type(grads) in {list, tuple}:
			outputs += grads
		else:
			outputs.append(grads)

		# [3]
		self.loss_and_grads = K.function([self.output_img], outputs)

	def style(self):
		"""
		Run L-BFGS over the pixels of the generated image so as to 
		minimize the neural style loss.
		"""
		print('\nDone initializing... Ready to style!')

		# initialize white noise image
		if K.image_data_format() == 'channels_first':
			x = np.random.uniform(0, 255, (1, 3, self.img_nrows, self.img_ncols)) - 128.
		else:
			x = np.random.uniform(0, 255, (1, self.img_nrows, self.img_ncols, 3)) - 128.

		for i in range(self.iterations):
			print('\n\tIteration: {}'.format(i+1))

			toc = time.time()
			x, min_val, info = fmin_l_bfgs_b(self.loss, x.flatten(), fprime=self.grads, maxfun=20)

			# save current generated image
			img = deprocess_image(x.copy(), self.img_nrows, self.img_ncols)
			fname = self.output_img_path + '_at_iteration_%d.png' % (i+1)
			imsave(fname, img)

			tic = time.time()

			print('\t\tImage saved as', fname)
			print('\t\tLoss: {:.2e}, Time: {} seconds'.format(float(min_val), float(tic-toc)))

	def loss(self, x):
		# reshape
		if K.image_data_format() == 'channels_first':
			x = x.reshape((1, 3, self.img_nrows, self.img_ncols))
		else:
			x = x.reshape((1, self.img_nrows, self.img_ncols, 3))

		outs = self.loss_and_grads([x])
		loss_value = outs[0]
		return loss_value

	def grads(self, x):
		# reshape
		if K.image_data_format() == 'channels_first':
			x = x.reshape((1, 3, self.img_nrows, self.img_ncols))
		else:
			x = x.reshape((1, self.img_nrows, self.img_ncols, 3))

		outs = self.loss_and_grads([x])

		if len(outs[1:]) == 1:
			grad_values = outs[1].flatten().astype('float64')
		else:
			grad_values = np.array(outs[1:]).flatten().astype('float64')
		return grad_values
