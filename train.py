import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda
import sys
import argparse
import os
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import adam
from keras.preprocessing.image import ImageDataGenerator
import json
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from time import time
import logging
import copy
import math
import random
from keras import backend as k

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
    log_lines = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            log_lines.append(line)
    return log_lines


def get_image_and_measurement(line, old_root=None, new_root=None):
    """

    :param line:
    :param old_root:
    :param new_root:
    :return:
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


def create_model(units=1, loss_function='mse', input_shape=(160, 320, 3), gpus=1, learning_rate=0.001):
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
    conv_model = Sequential()

    # Find the color space
    conv_model.add(Convolution2D(3, 1, 1, input_shape=input_shape))

    conv_model.add(Convolution2D(32, 3, 3))
    conv_model.add(Convolution2D(32, 3, 3))
    conv_model.add(Activation('relu'))
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))
    conv_model.add(Dropout(0.5))

    conv_model.add(Convolution2D(64, 3, 3))
    conv_model.add(Convolution2D(64, 3, 3))
    conv_model.add(Activation('relu'))
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))
    conv_model.add(Dropout(0.5))

    conv_model.add(Convolution2D(128, 3, 3))
    conv_model.add(Convolution2D(128, 3, 3))
    conv_model.add(Activation('relu'))
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))
    conv_model.add(Dropout(0.5))

    conv_model.add(Activation('relu'))
    conv_model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    conv_model.add(Flatten())
    conv_model.add(Dense(512))
    conv_model.add(Dense(64))
    conv_model.add(Dense(16))
    conv_model.add(Dense(units))

    opt = adam(lr=learning_rate)

    if gpus <= 1:
        conv_model.compile(loss=loss_function, optimizer=opt, metrics=['accuracy'])
    else:
        gpu_list = []
        [gpu_list.append('gpu(%d)' % i) for i in range(gpus)]
        conv_model.compile(loss=loss_function, optimizer=opt, metrics=['accuracy'],
                           context=gpu_list)
    return conv_model


def custom_get_params(self):
    """
    Function to patch issue in Keras
    :param self: Sci-kit parameters.
    :return: Deep copy of the parameters.
    """
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res


def get_all_images_and_measurements(line_tuples, old_root=None, new_root=None):
    """
    Retrieves individual images and measurements from all log files.
    :param line_tuples:
    :param old_root:
    :param new_root:
    :return:
    """
    all_center_images = []
    all_left_images = []
    all_right_images = []
    all_measurements = []

    for record in line_tuples:  # Each log file
        current_lines = record[1]  # Each line in the CSV - 0 would be log file path
        for line in current_lines:
            center_image, left_image, right_image, measurement = get_image_and_measurement(line,
                                                                                           old_root,
                                                                                           new_root)
            all_center_images.append(center_image)
            all_left_images.append(left_image)
            all_right_images.append(right_image)
            all_measurements.append(measurement)

    return np.array(all_center_images), np.array(all_left_images), \
           np.array(all_right_images), np.array(all_measurements)


def augment_brightness_camera_images(image):
    """
    Note: Cited from blog post https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9,
    which was a recommended reading by my mentor. Function used to augment my dataset to
    improve model performance.
    :param image: Image file opened by OpenCV
    :return:
    """
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1[:, :, 2][image1[:, :, 2] > 255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def adjust_side_images(measurement_value, adjustment_offset, side):
    """
    Implementation of usage of left and right images to simulate edge correction,
    as suggested in blog post by Vivek Yadav,
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9,
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
    translation_x = translation_range * np.random.uniform() - translation_range / 2
    translated_steering_angle = steering_angle + translation_x / translation_range * 2 * .2
    translation_y = 40 * np.random.uniform() - 40 / 2
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
    shadow_mask = 0 * image_hls[:, :, 1]
    x_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]

    shadow_mask[((x_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (y_m - top_y) >= 0)] = 1
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright

    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

    return image


def flip_image_and_measurement(image, measurement):
    """
    Flips image so it looks as though it was made going from the opposite direction.

    Note: Implementation of usage of left and right images to simulate edge correction,
    as suggested in blog post by Vivek Yadav,
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9,
    as suggested reading by my mentor, Rahul. Function used to augment my dataset to improve
    model performance.
    :param image:
    :param measurement:
    :return:
    """
    return cv2.flip(image, 1), measurement * -1


def crop_image(image, horizon_divisor, hood_pixels, crop_height, crop_width):
    """
    Note: Cited and refactored from blog post
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9,
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
    image = image[math.floor(shape[0] / horizon_divisor):shape[0] - hood_pixels, 0:shape[1]]
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


def full_augment_image(image, position, measurement, shift_offset=0.004):
    """

    :param image:
    :param position:
    :param measurement:
    :param shift_offset:
    :return:
    """
    aug_images = []
    aug_measurements = []

    measurement = adjust_side_images(measurement, .25, position)

    bright = augment_brightness_camera_images(image)
    shadow = add_random_shadow(image)
    flipped, flipped_mmt = flip_image_and_measurement(image, measurement)

    aug_images.append(image)
    aug_measurements.append(measurement)
    aug_images.append(bright)
    aug_measurements.append(measurement)
    aug_images.append(shadow)
    aug_measurements.append(measurement)
    aug_images.append(flipped)
    aug_measurements.append(flipped_mmt)

    if position == 'center':
        translated, shift_mmt = shift_image_position(image, measurement, shift_offset)
        bright_shifted, bright_shift_mmt = shift_image_position(bright, measurement, shift_offset)
        shadow_shifted, shadow_shift_mmt = shift_image_position(shadow, measurement, shift_offset)
        flipped_shifted, flipped_shift_mmt = shift_image_position(flipped, flipped_mmt, shift_offset)

        aug_images.append(translated)
        aug_measurements.append(shift_mmt)

        aug_images.append(bright_shifted)
        aug_measurements.append(bright_shift_mmt)

        aug_images.append(shadow_shifted)
        aug_measurements.append(shadow_shift_mmt)

        aug_images.append(flipped_shifted)
        aug_measurements.append(flipped_shift_mmt)

    # So, for non-center, will have 4, and if center, will have 8. So, 12 images per row.
    return aug_images, aug_measurements


def shuffle_data(shuffle_input_images, shuffle_input_measurements):
    """

    :param shuffle_input_images:
    :param shuffle_input_measurements:
    :return: shuffled_images, shuffled_measurements
    """
    shuffled_images = []
    shuffled_measurements = []

    index_list = range(len(shuffle_input_measurements))
    shuffled_indexes = random.sample(index_list, len(index_list))

    for i in shuffled_indexes:
        shuffled_images.append(shuffle_input_images[i])
        shuffled_measurements.append(shuffle_input_measurements[i])

    return shuffled_images, shuffled_measurements


def select_and_augment_image_for_image_generator(center_image, left_image, right_image, measurement):
    """

    :param center_image:
    :param left_image:
    :param right_image:
    :param measurement:
    :return:
    """
    # Select image position
    image_position = np.random.randint(3)
    if image_position == 0:
        position = 'center'
    elif image_position == 1:
        position = 'left'
    else:
        position = 'right'

    # Adjust measurement if necessary.
    measurement = adjust_side_images(measurement, .25, position)

    # Set image
    if position == 'center':
        image = center_image
    else:
        if position == 'left':
            image = left_image
        else:
            image = right_image

    # Select transformation type
    transformation_selection = np.random.randint(3)

    # Switch statement for augmentation selection
    if transformation_selection == 0:
        return image, measurement
    elif transformation_selection == 1:
        return augment_brightness_camera_images(image), measurement
    else:
        return add_random_shadow(image), measurement


def generate_and_augment_training_data_by_batch(gen_center_images,
                                                gen_left_images,
                                                gen_right_images,
                                                gen_measurements,
                                                height=64, channels=3, width=64, batch_size=32):
    """

    :param gen_center_images:
    :param gen_left_images:
    :param gen_right_images:
    :param gen_measurements:
    :param height:
    :param channels:
    :param width:
    :param batch_size:
    :return:
    """
    logger.info("In generate: All_left: " + str(len(gen_left_images)))
    while 1:
        batch_images = np.zeros((batch_size, height, width, channels))
        batch_measurements = np.zeros(batch_size)

        for batch_index in range(0, ((len(measurements) // batch_size) - 1)):
            starting_index_for_batch = batch_index * batch_size
            ending_index_for_batch = starting_index_for_batch + batch_size
            logger.info("Batch_index: " + str(batch_index))
            logger.info("Starting index: " + str(starting_index_for_batch))
            logger.info("Ending index: " + str(ending_index_for_batch))

            batch_center = gen_center_images[starting_index_for_batch:ending_index_for_batch]
            batch_left = gen_left_images[starting_index_for_batch:ending_index_for_batch]
            batch_right = gen_right_images[starting_index_for_batch:ending_index_for_batch]
            batch_measurements = gen_measurements[starting_index_for_batch:ending_index_for_batch]

            final_batch_images = []
            final_batch_measurements = []

            # Augment the images
            logger.info("Batch_mmt: " + str(len(batch_measurements)))
            logger.info("Left: " + str(len(batch_left)))
            for i in range(len(batch_measurements)):
                index_images = []
                index_measurements = []

                gen_center_images, gen_center_measurements = full_augment_image(batch_center[i], 'center',
                                                                                batch_measurements[i])
                [index_images.append(c_img) for c_img in gen_center_images]
                [index_measurements.append(c_mmt) for c_mmt in gen_center_measurements]

                gen_left_images, gen_left_measurements = full_augment_image(batch_left[i], 'left',
                                                                            batch_measurements[i])
                [index_images.append(l_img) for l_img in gen_left_images]
                [index_measurements.append(l_mmt) for l_mmt in gen_left_measurements]

                gen_right_images, gen_right_measurements = full_augment_image(batch_right[i], 'right',
                                                                              batch_measurements[i])
                [index_images.append(r_img) for r_img in gen_right_images]
                [index_measurements.append(r_mmt) for r_mmt in gen_right_measurements]

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
    # lines = get_log_lines(config['input_path'])
    logger.info("Getting log lines...")
    log_paths = get_file_list(config['input_path'])
    lines = []
    [lines.append([path, get_log_lines(path)]) for path in log_paths]

    center_images, left_images, right_images, measurements = \
        get_all_images_and_measurements(lines, old_root=config['old_image_root'],
                                        new_root=config['new_image_root'])
    center_images = [crop_image(img, horizon_divisor=5, hood_pixels=25,
                                crop_height=64, crop_width=64) for img in center_images]
    left_images = [crop_image(img, horizon_divisor=5, hood_pixels=25,
                              crop_height=64, crop_width=64) for img in left_images]
    right_images = [crop_image(img, horizon_divisor=5, hood_pixels=25,
                               crop_height=64, crop_width=64) for img in right_images]

    # Augment and select an individual vantage point.
    selected_images = []
    selected_measurements = []

    for idx in range(len(measurements)):
        img, mmt = select_and_augment_image_for_image_generator(center_images[idx],
                                                                left_images[idx],
                                                                right_images[idx],
                                                                measurements[idx])
        selected_images.append(img)
        selected_measurements.append(mmt)

    logger.info("Selected images: " + str(len(selected_images)))
    logger.info("Selected measurements: " + str(len(selected_measurements)))

    # Shuffle the data once
    images, measurements = shuffle_data(selected_images, selected_measurements)

    # Train/Test Split
    validation_index = int(len(measurements) * (1 - config['test_size']))
    images_test = np.array(images[-validation_index:])
    measurements_test = np.array(measurements[-validation_index:])
    images_train = np.array(images[:-validation_index])
    measurements_train = np.array(measurements[:-validation_index])
    logger.info("Validation set of length " + str(len(measurements_test)))
    logger.info("Training set of length " + str(len(measurements_train)))

    model = create_model(config['units'], gpus=config['gpus'], input_shape=(64, 64, 3),
                         learning_rate=config['learning_rate'])
    ckpt_path = config['checkpoint_path'] + "/augment_simple_{epoch:02d}_{val_acc:.2f}.hdf5"
    checkpointer = ModelCheckpoint(ckpt_path, verbose=1, save_best_only=True)

    # Establish tensorboard
    if config["use_tensorboard"] == "True":
        tensorboard = TensorBoard(log_dir=config["tensorboard_log_dir"] + "/{}".format(time()), histogram_freq=1,
                                  write_graph=True)
        callbacks = [checkpointer, tensorboard]
    else:
        callbacks = [checkpointer]

    # model.fit(X_train, y_train, nb_epoch=config['epochs'], batch_size=config['batch_size'],
    #          validation_split=0.2, shuffle=True, callbacks=callbacks)
    logger.info("Training the model...")

    train_datagen = ImageDataGenerator(
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(images_train, measurements_train,
                                         batch_size=config['batch_size'])

    validation_generator = test_datagen.flow(images_test, measurements_test,
                                             batch_size=config['batch_size'])

    model.fit_generator(train_generator, samples_per_epoch=8, nb_epoch=config['epochs'],
                        validation_data=validation_generator, nb_val_samples=2, callbacks=callbacks)

    if config['output_path'].endswith('.h5'):
        model.save(config['output_path'])
    else:
        model.save(config['output_path'] + '.h5')

    k.clear_session()
    sys.exit(0)
