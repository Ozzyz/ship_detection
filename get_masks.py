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
	empty_images = []
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

		if np.array_equal(all_masks, np.zeros((768, 768))):
			empty_images.append(imageID)
			os.remove(data_folder + "/" + imageID)
			continue

		all_all_masks[imageID] = all_masks
		imsave(dst_folder + imageID, all_masks)

	num_boats = len(images) - counter
	print("Empty imageS: ", empty_images)
	print(f"Number of images containing boats: {len(images)- len(empty_images)}")
	print(f"Number of images not containing boats: {len(empty_images)}")

def gen_all_masks():
	# NOTE: Assumes that masks folders exists
	write_masks('../small_test_segmentations.csv', 'data/test_images/test_small', 'data/test_masks/test_masks_small/')
	#write_masks('../small_train_segmentations.csv', 'data/images/train_small', 'data/masks/train_masks_small/')
	write_masks('small_validation_set.csv', 'data/val_images/val_small/', 'data/val_masks/val_masks_small/')
	pass
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
