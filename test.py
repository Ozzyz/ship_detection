from model import basic_cnn, hourglass_cnn, unet, ship_cnn, test_unet
from keras.datasets import mnist
from keras import utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
from skimage.data import imread
import numpy as np
import matplotlib.pyplot as plt


def load_data_generator(train_folderpath, mask_folderpath, img_size=(768, 768), mask_size=(768, 768), batch_size=32):
	"""
	Returns a data generator with masks and training data specified by the directory paths given.
	"""
	data_gen_args = dict(
		rescale=1. / 255,
		rotation_range=90,
		width_shift_range=0.1,
		height_shift_range=0.1,
		zoom_range=0.2)
	image_datagen = ImageDataGenerator(**data_gen_args)
	mask_datagen = ImageDataGenerator(**data_gen_args)

	seed = 42

	image_generator = image_datagen.flow_from_directory(train_folderpath, class_mode=None,
														target_size=img_size, seed=seed, color_mode='rgb',
														batch_size=batch_size)
	mask_generator = mask_datagen.flow_from_directory(mask_folderpath, class_mode=None,
													  target_size=mask_size, seed=seed, color_mode='grayscale',
													  batch_size=batch_size)

	return zip(image_generator, mask_generator)


def run():
	input_dim = (768, 768, 3)
	print("Instantiating model...")
	model = ship_cnn(input_dim)
	print(model.summary())
	print("Creating training generator...")
	#val_generator = load_data_generator('/data/val_images', 'data/val_masks', batch_size=2)
	train_generator = load_data_generator('data/trainX', 'data/trainY', batch_size=1)
	print("Fitting model to generator")
	model.fit_generator(train_generator, steps_per_epoch=10, epochs=20)#, validation_data=val_generator)
	model.save('test_model.h5')


def apply_pretrained_model(filepath, data_folder, mask_folder):
	"""
		Loads a pretrained keras model and applies it to a selection of the test images, creating a
		submission csv.
	"""
	model = load_model(filepath)

	#image_names = os.listdir(data_folder)
	image_names = ["00a52cd2a.jpg", "00abc623a.jpg"]
	_, axis = plt.subplots(1, 2)

	for (ax1, ax2), c_img_name in zip(axis, image_names):
		c_path = os.path.join(data_folder, c_img_name)
		c_img = imread(c_path)
		first_img = np.expand_dims(c_img, 0) / 255.0
		first_seg = model.predict(first_img)
		ax1.imshow(first_img[0])
		ax1.set_title('Image')
		ax2.imshow(first_seg[0, :, :, 0], vmin=0, vmax=1)
		ax2.set_title('Prediction')

	plt.show()


if __name__ == '__main__':
	run()
	#apply_pretrained_model('test_model.h5', 'data/trainX/train_small', 'data/trainY/train_masks')

