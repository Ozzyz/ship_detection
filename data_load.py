
from keras.datasets import mnist, cifar10
from keras import utils
import pandas as pd
import shutil
from os import mkdir
def load_mnist():
    """
        Load the mnist dataset (60000 28x28 grayscale images)
    """
    num_classes = 10
    img_rows, img_cols = (28, 28)
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, y_train = prepare_image_dataset(x_train, y_train, (img_rows, img_cols), num_classes)
    x_test, y_test = prepare_image_dataset(x_train, y_train, (img_rows, img_cols), num_classes)
    input_shape = (img_rows, img_cols, 1)
    return (x_train, y_train), (x_test, y_test), input_shape


def load_cifar10():
    """
        Loads the cifar10 dataset (60000 32x32 colur images) in 10 balanced classes.
    """
    num_classes = 10
    img_rows, img_cols = (32, 32)
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train, y_train = prepare_image_dataset(x_train, y_train, (img_rows, img_cols), num_classes, reshape=False)
    x_test, y_test = prepare_image_dataset(x_train, y_train, (img_rows, img_cols), num_classes, reshape=False)
    input_shape = (img_rows, img_cols, 3)
    return (x_train, y_train), (x_test, y_test), input_shape

    
def prepare_image_dataset(x_data, y_data, img_dim, num_classes, reshape=True):
    """
        Normalizes the images in the datasets and converts the labels to categoricals.
    """
    img_rows, img_cols = img_dim
    if reshape:
        x_data = x_data.reshape(x_data.shape[0], img_rows, img_cols)
    x_data = x_data.astype('float32')/255
    # convert class vectors to binary class matrices
    y_data = utils.to_categorical(y_data, num_classes)
    
    return (x_data, y_data)


def load_small_dataset(csv_filepath, data_folder, dst_folder):
    # Takes input a csv, and loads all files specified in the csv into dst_folder
    # NOTE: Assumes that dst_folder exists!
    df = pd.read_csv(csv_filepath)
    for filename in df['ImageId']:
        shutil.copy2(data_folder + filename, dst_folder + filename)


#load_small_dataset('../small_train_segmentations.csv', data_folder="../train/", dst_folder = "../train_small/")
#load_small_dataset('../small_test_segmentations.csv', data_folder="../test/", dst_folder = "../test_small/")
#load_small_dataset('small_validation_set.csv', data_folder='../train/', dst_folder='./val_small/')