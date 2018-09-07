import pandas as pd
import numpy as np
import os
from skimage.data import imread
from skimage.io import imsave
import matplotlib.pyplot as plt
import warnings

#warnings.simplefilter("ignore")


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
	'''
	mask_rle: run-length as string formated (start length)
	shape: (height,width) of array to return
	Returns numpy array, 1 - mask, 0 - background
	'''
	s = mask_rle.split()
	starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
	starts -= 1
	ends = starts + lengths
	img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
	for lo, hi in zip(starts, ends):
		img[lo:hi] = 1
	return img.reshape(shape).T  # Needed to align to RLE direction


def write_masks(masks_filepath, data_folder, dst_folder):
	masks = pd.read_csv(masks_filepath)
	images = os.listdir(data_folder)
	all_all_masks = {}

	counter = 0

	for imageID in images:
		img_masks = masks.loc[masks['ImageId'] == imageID, 'EncodedPixels'].tolist()

		# Take the individual ship masks and create a single mask array for all ships
		all_masks = np.zeros((768, 768))
		for mask in img_masks:
			# Image has no masks - i.e no ships
			if mask is np.nan:
				counter += 1
				continue
			all_masks += rle_decode(mask)

		all_all_masks[imageID] = all_masks
		imsave(dst_folder + imageID, all_masks)

	num_boats = 1337 - counter

	print(f"Number of images containing boats: {num_boats}")
	print(f"Number of images not containing boats: {counter}")

def gen_all_masks():
	# NOTE: Assumes that masks folders exists
	#write_masks('../small_test_segmentations.csv', '../test_small', '../test_masks_small/')
	#write_masks('../small_train_segmentations.csv', '../train_small', '../train_masks_small/')
	#write_masks('small_validation_set.csv', 'val_small/', 'val_masks_small/')

if __name__ == '__main__':
	gen_all_masks()
"""
img = imread('data/train_small/00f34434e.jpg')

fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
axarr[0].axis('off')
axarr[1].axis('off')
axarr[2].axis('off')
axarr[0].imshow(img)
axarr[1].imshow(all_all_masks['00f34434e.jpg'])
axarr[2].imshow(img)
axarr[2].imshow(all_all_masks['00f34434e.jpg'], alpha=0.4)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()
"""
