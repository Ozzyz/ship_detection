from skimage.data import imread
from skimage.io import imshow, show
import os 
import numpy as np
import matplotlib.pyplot as plt

def data_generator(image_folder, mask_folder, batch_size = 32, transform_params = None, rescale=1./255):
    # Assumes images and masks have same filename but are located in different folders
    image_names = os.listdir(image_folder) 
    while True:
        for idx in range(0, len(image_names), batch_size):
            batch_x = []
            batch_y = []
            offset = min(idx + batch_size, len(image_names))

            image_batch = image_names[idx:offset]

            for name in image_batch:
                img_filepath = os.path.join(image_folder, name)
                mask_filepath = os.path.join(mask_folder, name)
                img = imread(img_filepath)*rescale
                augmented_image = augment_image(img)

                mask = imread(mask_filepath)
                augmented_mask = augment_image(mask)*rescale
                batch_x.append(augmented_image)
                batch_y.append(augmented_mask)
            yield np.array(batch_x), np.array(batch_y)


def augment_image(image):
    # TODO: Augment image
    return image 


if __name__ == '__main__':
    mask_folder = "data/masks/train_masks_small/"
    image_folder = "data/images/train_small/"
    datagen = data_generator(image_folder, mask_folder)
    plot_generator_images(datagen)


def plot_generator_images(generator, NUM_IMAGES):
    NUM_IMAGES = 5
    _, axis = plt.subplots(NUM_IMAGES, 2)

    for NUM, (ax1, ax2), (img, mask) in zip(range(NUM_IMAGES), axis, datagen):
        ax1.imshow(img[NUM, ...])
        ax2.imshow(mask[NUM, ...])
    plt.show()