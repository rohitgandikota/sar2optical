import numpy as np
from neural_styler import Neural_Styler
from keras import backend as K
from argparse import ArgumentParser
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# default arguments
CONVNET = 'vgg16'
CONTENT_WEIGHT = 7.5e0
    STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2
ITERATIONS = 100
CONTENT_LAYER = 'block4_conv2'
STYLE_LAYERS = ['block1_conv1',
				'block2_conv1',
				'block3_conv1', 
				'block4_conv1', 
				'block5_conv1']

def build_parser():
	
	# initialize command line parser
	description = 'Neural Style Implementation in Keras.'
	parser = ArgumentParser(description=description)

	parser.add_argument('base_img_path', metavar='base', type=str,
						help='path to base image.')
	parser.add_argument('style_img_path', metavar='style', type=str,
						help='path to style image.')
	parser.add_argument('output_img_path', metavar='output', type=str,
						help='path to output image.')
	parser.add_argument('--iters', type=int, default=ITERATIONS,
						metavar='iterations', help='Number of iterations.')
	parser.add_argument('--content_weight', type=float, default=CONTENT_WEIGHT,
						help='Weight for content feature loss')
	parser.add_argument('--style_weight', type=float, default=STYLE_WEIGHT,
						help='Weight for style feature loss')
	parser.add_argument('--tv_weight', type=float, default=TV_WEIGHT,
						help='Weight for total variation loss')
	parser.add_argument('--width', type=int, help='output image width')
	parser.add_argument('--convnet', type=str, default=CONVNET,
						help='VGG model (16 or 19)')
	return parser

def main():

	parser = build_parser()
	options = parser.parse_args()

	# grab convnet model
	convnet = options.convnet

	if convnet not in ('vgg16', 'vgg19'):
		raise ValueError('You have specified an incorrect VGG model (16 or 19 only).')

	# grab output img width
	output_width = options.width

	# grab paths
	base_img_path = options.base_img_path
	style_img_path = options.style_img_path
	output_img_path = options.output_img_path

	# grab learning params
	iterations = options.iters
	total_variation_weight = options.tv_weight
	style_weight = options.style_weight
	content_weight = options.content_weight

	# instantiate neural style object
	neural_styler = Neural_Styler(base_img_path,
				      style_img_path,
				      output_img_path,
				      output_width,
				      convnet, 
				      content_weight, 
				      style_weight, 
				      total_variation_weight,
				      CONTENT_LAYER,
				      STYLE_LAYERS,
				      iterations)

	# create style image
	neural_styler.style()

if __name__ == '__main__':
	main()
