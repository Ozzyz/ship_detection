from model import basic_cnn, hourglass_cnn, unet
import time
from keras.datasets import mnist
from keras import utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
from skimage.data import imread
import numpy as np
import matplotlib.pyplot as plt
import random
from loss_funcs import dice_coef, dice_p_bce, true_positive_rate

def load_data_generator(train_folderpath, mask_folderpath, img_size = (768, 768), mask_size=(768,768), batch_size=32):
    """
    Returns a data generator with masks and training data specified by the directory paths given.
    """
    data_gen_args = dict(featurewise_center=True,
                            featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     rescale=1./255,
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
    val_generator = load_data_generator('/data/val_images', 'data/val_masks', batch_size=2)
    train_generator = load_data_generator('data/images', 'data/masks', batch_size=2)
    print("Fitting model to generator")
    model.fit_generator(train_generator, steps_per_epoch=250, epochs=20, validation_data = val_generator, validation_steps=100)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model.save("trained_model_{0}.h5".format(timestr))


def apply_pretrained_model(filepath, data_folder, mask_folder):
    """
        Loads a pretrained keras model and applies it to a selection of the test images, creating a 
        submission csv. 
    """
    model = load_model(filepath, custom_objects={'dice_p_bce': dice_p_bce, 'true_positive_rate': true_positive_rate, 'dice_coef': dice_coef})
    NUM_IMAGES = 7
    
    image_names = random.sample(os.listdir(data_folder), NUM_IMAGES)
    ground_truths = random.sample(os.listdir(mask_folder), NUM_IMAGES)

    _, axis = plt.subplots(NUM_IMAGES, 3)

    for (ax1, ax2, ax3), c_img_name, mask_img_name in zip(axis, image_names, ground_truths):
        print("C_img,", c_img_name)
        print("Ground truth:", mask_img_name)
        c_path = os.path.join(data_folder, c_img_name)
        c_img = imread(c_path)
        first_img = np.expand_dims(c_img, 0)
        first_seg = model.predict(first_img)
        print(c_img.shape, c_img.min(), c_img.max(), c_img.mean())
        print(first_seg.shape, first_seg.min(), first_seg.max(), first_seg.mean())
        mask_path = os.path.join(mask_folder, mask_img_name)
        ground_truth = imread(mask_path)
        ground_truth =  np.expand_dims(ground_truth, 0)
        ax1.imshow(first_img[0])
        ax1.set_title('Image')
        ax2.imshow(first_seg[0, :, :, 0], vmin = 0, vmax = 1)
        ax2.set_title('Prediction')
        ax3.imshow(ground_truth[0, :, :], vmin=0, vmax=1)
        ax3.set_title('Ground truth')

    plt.show()
if __name__ == '__main__':
    #run()
    apply_pretrained_model('trained_model.h5', 'data/test_images/test_small', 'data/test_masks/test_masks_small')
