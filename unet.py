from model import basic_cnn, hourglass_cnn, unet
from keras.datasets import mnist
from keras import utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import os
from skimage.data import imread
import numpy as np

#train = os.listdir('data/train_small')
#test = os.listdir('data/test_small')
#train_masks = os.listdir('data/train_masks')
#test_masks = os.listdir('data/test_masks')

def load_data(filepath, num_channels):
	"""
	Returns all images in the specified folder filepath
	:param filepath: Name of folder where images are stored
	:return: An array of images
	"""
	data = np.empty((1337, 768, 768, num_channels), dtype=np.uint8)

	i = 0
	for item in os.listdir(filepath):
		if os.path.isfile(filepath + item):
			data[i] = imread(filepath + item).reshape(768, 768, num_channels)
	i += 1

	return data

def run():
	train = load_data('data/train_small/', 3)
	test = load_data('data/test_small/', 3)
	train_masks = load_data('data/train_masks/', 1)
	test_masks = load_data('data/test_masks/', 1)
	print('images loaded')

	datagen = ImageDataGenerator(
		featurewise_center=True,
		featurewise_std_normalization=True
	)

	datagen.fit(train)

	#print(train.shape)

	model = unet(input_dim=(768, 768, 3))
	print('model ready')

	model.fit_generator(datagen.flow(train, train_masks, batch_size=32), steps_per_epoch=len(train), epochs=10)
	#model.fit(train, train_masks, validation_data=(test, test_masks))

#run()

model = unet(input_dim=(768, 768, 3))