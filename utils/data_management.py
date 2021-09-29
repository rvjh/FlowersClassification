import os
import utils.config as config
import tensorflow as tf
import keras_preprocessing
import keras_preprocessing.image
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

def manage_input_data(input_image):
    """converting the input array into desired dimension

    Args:
        input_image (nd array): image nd array

    Returns:
        nd array: resized and updated dim image
    """
    images = input_image
    size = config.IMAGE_SIZE[:-1]
    resized_input_img = tf.image.resize(
        images,
        size
        )

    final_img = np.expand_dims(resized_input_img, axis=0)
    return final_img


def train_valid_data_gen(IMAGE_SIZE = config.IMAGE_SIZE[:-1]):

    training_dir = config.TRAIN_DIR
    validation_dir = config.VAL_DIR
    test_dir = config.TEST_DIR

    training_data_gen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    validation_data_gen = ImageDataGenerator(rescale=1. / 255)
    test_data_gen = ImageDataGenerator(rescale=1. / 255)

    training_set = training_data_gen.flow_from_directory(training_dir,
                                                         target_size=IMAGE_SIZE,
                                                         batch_size=32,
                                                         class_mode='categorical')
    validation_set = validation_data_gen.flow_from_directory(validation_dir,
                                                            target_size=IMAGE_SIZE,
                                                            batch_size=32,
                                                            class_mode='categorical')

    return training_set , validation_set












