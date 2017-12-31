import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda
import sys
import argparse
import os
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import adam
import json
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from time import time
import logging
import pickle
import copy
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args(arguments):
    """
    Parses arguments given at the command line.
    :param arguments: Arguments given at the command line
    :return: Dict of variables parsed from the arguments
    """
    parser = argparse.ArgumentParser(description="Trains a behavioral cloning model from a given training file set.")
    parser.add_argument('-c', '--configuration', help="File path configuration file", required=True,
                        dest='config')

    return vars(parser.parse_args(arguments))


def load_config(config_name):
    """
    loads a json config file and returns a config dictionary
    """
    with open(config_name) as config_file:
        configuration = json.load(config_file)
        return configuration

def get_file_list(dir_path):
    """
    Get list of files
    :param dir_path:
    :return: List of driving log files to open.
    """
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        [file_list.append(os.path.join(root, file)) for file in files if file.endswith('.csv')]
    return file_list

def get_log_lines(path):
    """
    Gets list of records from driving log.
    :param path: Input path for driving log.
    :return: List of driving log records.
    """
    lines = []
    #with open(path + '/driving_log.csv') as csvfile:
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def get_image_and_measurement(line, old_root=None, new_root=None):
    """

    :param line:
    :param old_root:
    :param new_root:
    :param image_position: Image position within the file. Value can be 0 for center, 1 for left, or 2 for right.
    :return: image array and measurement
    """
    center_image_path = line[0]
    left_image_path = line[1]
    right_image_path = line[2]

    if new_root and old_root:
        center_image_path = center_image_path.replace(old_root, new_root)
        left_image_path = left_image_path.replace(old_root, new_root)
        right_image_path = right_image_path.replace(old_root, new_root)
    center_image = cv2.imread(center_image_path)
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)
    measurement = float(line[3])

    return center_image, left_image, right_image, measurement

def create_model(units=1, loss_function='mse', optimizer='adam', input_shape=(160,320,3), gpus=1, learning_rate=0.001):
    """
    Constructs Keras model object
    :return: Compiled Keras model object
    """
    """
    # ORIGINAL
    model = Sequential()
    model.add(Convolution2D(160, 3, 3, input_shape=input_shape))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))

    model.add(Activation('relu'))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(units))
    """
    model = Sequential()

    # Find the color space
    model.add(Convolution2D(3, 1, 1, input_shape=input_shape))

    model.add(Convolution2D(32, 3, 3))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Activation('relu'))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    model.add(Flatten())
    #model.add(Dense(512))
    #model.add(Dense(64))
    model.add(Dense(units))

    # TODO: Fix signature so multiple optimizers can be accepted.
    opt = adam(lr=learning_rate)

    if gpus <= 1:
        model.compile(loss=loss_function, optimizer=opt, metrics=['accuracy'])
    else:
        gpu_list = []
        [gpu_list.append('gpu(%d)' % i) for i in range(gpus)]
        model.compile(loss=loss_function, optimizer=opt, metrics=['accuracy'],
                      context=gpu_list)
    return model

def custom_get_params(self, **params):
    """
    Function to patch issue in Keras
    :param self:
    :param params:
    :return:
    """
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res


def get_all_images_and_measurements(line_tuples, old_root=None, new_root=None):
    """
    Retrieves individual images and measurements from all log files.
    :param line_tuples: Line tuples extracted from log files. Each record:
    Log file, [lines of log file].
    :return: Images and measurements for all files.
    """
    all_center_images = []
    all_left_images = []
    all_right_images = []
    all_measurements = []

    for record in line_tuples: # Each log file
        current_lines = record[1] # Each line in the CSV - 0 would be log file path
        for line in current_lines:
            center_image, left_image, right_image, measurement = get_image_and_measurement(line, old_root, new_root)
            all_center_images.append(center_image)
            all_left_images.append(left_image)
            all_right_images.append(right_image)
            all_measurements.append(measurement)

    return np.array(all_center_images), np.array(all_left_images), np.array(all_right_images), np.array(all_measurements)

def augment_brightness_camera_images(image):
    """
    Note: Cited from blog post https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9,
    which was a recommended reading by my mentor. Function used to augment my dataset to
    improve model performance.
    :param image: Image file opened by OpenCV
    :return:
    """
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def adjust_side_images(measurement_value, adjustment_offset, side):
    """
    Implementation of usage of left and right images to simulate edge correction,
    as suggested in blog post by Vivek Yadav, https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9,
    as suggested reading by my mentor, Rahul. Function used to augment my dataset to improve
    model performance.
    :param measurement_value:
    :param adjustment_offset:
    :param side:
    :return:
    """
    if side == 'left':
        return measurement_value + adjustment_offset
    elif side == 'right':
        return measurement_value - adjustment_offset
    elif side == 'center':
        return measurement_value

def shift_image_position(image, steering_angle, translation_range):
    """
    Note: Cited from blog post https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9,
    which was a recommended reading by my mentor. Function used to augment my dataset to
    improve model performance.
    :param image:
    :param steering_angle:
    :param translation_range:
    :return: translated_image, translated_steering_angle
    """
    translation_x = translation_range * np.random.uniform() - translation_range/2
    translated_steering_angle = steering_angle + translation_x/translation_range*2*.2
    translation_y = 40 * np.random.uniform() - 40/2
    translation_m = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rows = image.shape[0]
    cols = image.shape[1]
    translated_image = cv2.warpAffine(image, translation_m, (cols, rows))

    return translated_image, translated_steering_angle

def add_random_shadow(image):
    """
    Adding a random shadow mask to the image.
    Note: Cited from blog post https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9,
    which was a recommended reading by my mentor. Function used to augment my dataset to
    improve model performance.
    :param image: Image to add a shadow too.
    :return: Image with a random shadow added.
    """
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]

    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright

    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

    return image


def flip_image_and_measurement(image, measurement):
    """
    Flips image so it looks as though it was made going from the opposite direction.

    Note: Implementation of usage of left and right images to simulate edge correction,
    as suggested in blog post by Vivek Yadav, https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9,
    as suggested reading by my mentor, Rahul. Function used to augment my dataset to improve
    model performance.
    :param image:
    :return:
    """
    return cv2.flip(image, 1), measurement*-1

def crop_image(image, horizon_divisor, hood_pixels, crop_height, crop_width):
    """
    Note: Cited and refactored from blog post https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9,
    which was a recommended reading by my mentor. Function used to augment my dataset to
    improve model performance.
    :param image:
    :param horizon_divisor:
    :param hood_pixels:
    :param crop_height:
    :param crop_width:
    :return:
    """
    shape = image.shape
    image = image[math.floor(shape[0]/horizon_divisor):shape[0]-hood_pixels, 0:shape[1]]
    image = cv2.resize(image, (crop_width, crop_height))

    return image

def pick_random_vantage_point():
    """
    Pick left, center, or right at random.
    :return:
    """
    random_integer = np.random.randint(3)
    if random_integer == 0:
        return 'left'
    elif random_integer == 1:
        return 'right'
    else:
        return 'center'


def augment_image(image, position, measurement, shift_offset=0.004):
    """

    :param center_image:
    :param left_image:
    :param right_image:
    :param measurement:
    :return:
    """
    images = []
    measurements = []

    measurement = adjust_side_images(measurement, .25, position)

    #bright = augment_brightness_camera_images(image)
    #shadow = add_random_shadow(image)
    flipped, flipped_mmt = flip_image_and_measurement(image, measurement)

    images.append(image)
    measurements.append(measurement)
    #images.append(bright)
    #measurements.append(measurement)
    #images.append(shadow)
    #measurements.append(measurement)
    images.append(flipped)
    measurements.append(flipped_mmt)


    if position == 'center':
        translated, shift_mmt = shift_image_position(image, measurement, shift_offset)
        #bright_shifted, bright_shift_mmt = shift_image_position(bright, measurement, shift_offset)
        #shadow_shifted, shadow_shift_mmt = shift_image_position(shadow, measurement, shift_offset)
        flipped_shifted, flipped_shift_mmt = shift_image_position(flipped, flipped_mmt, shift_offset)

        images.append(translated)
        measurements.append(shift_mmt)

        #images.append(bright_shifted)
        #measurements.append(bright_shift_mmt)

        #images.append(shadow_shifted)
        #measurements.append(shadow_shift_mmt)

        images.append(flipped_shifted)
        measurements.append(flipped_shift_mmt)

    # So, for non-center, will have 4, and if center, will have 8. So, 12 images per row.
    return images, measurements

def generate_training_data_by_batch(center_images, left_images, right_images, measurements,
                                    height=64, channels=3, width=64, batch_size=32):
    """

    :param center_images:
    :param left_images:
    :param right_images:
    :param measurements:
    :param height:
    :param channels:
    :param width:
    :param batch_size:
    :return:
    """
    logger.info("In generate: All_left: " + str(len(left_images)))
    while 1:
        batch_images = np.zeros((batch_size, height, width, channels))
        batch_measurements = np.zeros(batch_size)

        # TODO: Figure out, do I shuffle beforehand? Also, do I... yeah, do I count the batch size as part of the images I've got?
        for batch_index in range(0, ((len(measurements) // batch_size) - 1)):
            starting_index_for_batch = batch_index * batch_size
            ending_index_for_batch = starting_index_for_batch + batch_size
            logger.info("Batch_index: " + str(batch_index))
            logger.info("Starting index: " + str(starting_index_for_batch))
            logger.info("Ending index: " + str(ending_index_for_batch))

            batch_center = center_images[starting_index_for_batch:ending_index_for_batch]
            batch_left = left_images[starting_index_for_batch:ending_index_for_batch]
            batch_right = right_images[starting_index_for_batch:ending_index_for_batch]
            batch_measurements = measurements[starting_index_for_batch:ending_index_for_batch]

            final_batch_images = []
            final_batch_measurements = []

            # Augment the images
            logger.info("Batch_mmt: " + str(len(batch_measurements)))
            logger.info("Left: " + str(len(batch_left)))
            for i in range(len(batch_measurements)):
                index_images = []
                index_measurements = []

                center_images, center_measurements = augment_image(batch_center[i], 'center', batch_measurements[i])
                [index_images.append(img) for img in center_images]
                [index_measurements.append(mmt) for mmt in center_measurements]

                left_images, left_measurements = augment_image(batch_left[i], 'left', batch_measurements[i])
                [index_images.append(img) for img in left_images]
                [index_measurements.append(mmt) for mmt in left_measurements]

                right_images, right_measurements = augment_image(batch_right[i], 'right', batch_measurements[i])
                [index_images.append(img) for img in right_images]
                [index_measurements.append(mmt) for mmt in right_measurements]

                # Pick a random image out of each of the augmented images. Keeps the batch size in check.
                random_index = np.random.randint(len(index_measurements))

                final_batch_images.append(index_images[random_index])
                final_batch_measurements.append(index_measurements[random_index])

        yield batch_images, batch_measurements


if __name__ == '__main__':

    # Set TensorFlow logging so it isn't so verbose.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Load the necessary parameters
    args = parse_args(sys.argv[1:])
    config = load_config(args['config'])

    # Load data
    #lines = get_log_lines(config['input_path'])
    logger.info("Getting log lines...")
    log_paths = get_file_list(config['input_path'])
    lines = []
    [lines.append([path, get_log_lines(path)]) for path in log_paths]

    logger.info("Getting all images and measurements..." + str(len(lines)))
    center_images, left_images, right_images, measurements = \
        get_all_images_and_measurements(lines, old_root=config['old_image_root'],
                                                           new_root=config['new_image_root'])
    center_images = [crop_image(img, horizon_divisor=5, hood_pixels=25,
                                                     crop_height=64, crop_width=64) for img in center_images]
    left_images = [crop_image(img, horizon_divisor=5, hood_pixels=25,
                                                     crop_height=64, crop_width=64) for img in left_images]
    right_images = [crop_image(img, horizon_divisor=5, hood_pixels=25,
                                                     crop_height=64, crop_width=64) for img in right_images]

    # Designate validation
    validation_index = int(len(measurements) * .2)
    center_images_valid = center_images[-validation_index:]
    left_images_valid = left_images[-validation_index:]
    right_images_valid = right_images[-validation_index:]
    measurements_valid = measurements[-validation_index:]

    center_images = center_images[:-validation_index]
    left_images = left_images[:-validation_index]
    right_images = right_images[:-validation_index]
    measurements = measurements[:-validation_index]
    logger.info("Validation set of length " + str(len(measurements_valid)))
    logger.info("Training set of length " + str(len(measurements)))

    model = create_model(config['units'], gpus=config['gpus'], input_shape=(64, 64, 3),
                         learning_rate=config['learning_rate'])
    ckpt_path = "/output/floyd_model_augment_2_{epoch:02d}_{val_acc:.2f}.hdf5"
    checkpointer = ModelCheckpoint(ckpt_path, verbose=1, save_best_only=True)

    # Establish tensorboard
    if config["use_tensorboard"] == "True":
        tensorboard = TensorBoard(log_dir=config["tensorboard_log_dir"] + "/{}".format(time()), histogram_freq=1,
                                  write_graph=True)
        callbacks = [checkpointer, tensorboard]
    else:
        callbacks = [checkpointer]

    #model.fit(X_train, y_train, nb_epoch=config['epochs'], batch_size=config['batch_size'],
    #          validation_split=0.2, shuffle=True, callbacks=callbacks)
    logger.info("Training the model...")
    train_gen = generate_training_data_by_batch(center_images, left_images, right_images, measurements,
                                                        batch_size=config['batch_size'])
    #valid_gen = generate_training_data_by_batch(center_images_valid, left_images_valid, right_images_valid,
                                                #measurements_valid, batch_size=1)
    model.fit_generator(train_gen, samples_per_epoch=(len(center_images) // config['batch_size']),
                        nb_epoch=config['epochs']) #, validation_data=valid_gen, nb_val_samples=4)

    if config['output_path'].endswith('.h5'):
        model.save(config['output_path'])
    else:
        model.save(config['output_path'] + '.h5')

    sys.exit(0)