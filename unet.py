from model import basic_cnn, hourglass_cnn, unet
from keras.datasets import mnist
from keras import utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import os
from skimage.data import imread
import numpy as np



def load_data_generator(train_folderpath, mask_folderpath, img_size = (768, 768), mask_size=(768,768), batch_size=32):
	"""
	Returns a data generator with masks and training data specified by the directory paths given.
	"""
	data_gen_args = dict(
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
						zoom_range=0.2)
	image_datagen = ImageDataGenerator(**data_gen_args)
	mask_datagen = ImageDataGenerator(**data_gen_args)

	seed = 42
	
	image_generator = image_datagen.flow_from_directory(train_folderpath, class_mode=None,
		target_size = img_size, seed=seed, color_mode = 'rgb', batch_size=batch_size)
	mask_generator = mask_datagen.flow_from_directory(mask_folderpath, class_mode=None, 
		target_size = mask_size,seed=seed, color_mode='grayscale', batch_size=batch_size)

	return zip(image_generator, mask_generator)



def run():
	input_dim = (768, 768, 3)
	print("Instantiating model...")
	model = unet(input_dim)
	print(model.summary())
	print("Creating training generator...")
	train_generator = load_data_generator('data/images', 'data/masks', batch_size=2)
	print("Fitting model to generator")
	model.fit_generator(train_generator, steps_per_epoch=10, epochs=50)
	model.save('trained_model.h5')
if __name__ == '__main__':
	run()