#!/usr/bin/env python
# encoding: utf-8

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.image as mpimg


CORRECTION = 0.2
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_csv(file):
    """
    Load training data
    """
    data_df = pd.read_csv(file)
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    return X, y


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :]


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def pick_image(center_image, left_image, right_image, angle):
    """
    Randomly choose an image from the center, left or right
    """
    images = (center_image, left_image, right_image)
    angles = (angle, angle+CORRECTION, angle-CORRECTION)

    i = np.random.choice(3)
    return images[i], angles[i]


def random_flip(image, angle):
    """
    Randomly flip the image to left or right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        angle = -angle
    return image, angle


def random_brightness(image, ratio=0.5):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
    brightness = np.float64(hsv[:, :, 2])
    brightness = brightness * (1.0 + np.random.uniform(-ratio, ratio))
    brightness[brightness>255] = 255
    brightness[brightness<0] = 0
    hsv[:, :, 2] = brightness
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return image


def random_translation(image, angle, range_x=100, range_y=10):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))

    return image, angle


def augment(center_image, left_image, right_image, angle):
    """
    Augment images through flip, shift and brightness tuning
    """
    # 1. randomly pick up a image
    image, angle = pick_image(center_image, left_image, right_image, angle)

    # 2. add flip image in order to recognize both clockwise and counter-clockwise roads
    # image, angle = random_flip(image, angle)

    # 3. randomly adjust shift
    image, angle = random_translation(image, angle)

    # 4. randomly adjust brightness
    image = random_brightness(image)

    return image, angle


def load_images(img_dir, img_names):
    return [mpimg.imread(os.path.join(img_dir, img_names[i].strip().split('/')[-1])) for i in range(3)]


def batch_generator1(img_dir, samples, batch_size=40, is_training=True):
    num_samples = len(samples)
    batch_images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    batch_angles = np.empty(batch_size)

    while True:
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            for i, batch_sample in enumerate(batch_samples):
                image_names = batch_sample[:3]
                angle = float(batch_sample[3])
                center_image, left_image, right_image = load_images(img_dir, image_names)
                if is_training and np.random.rand() < 0.6:
                    image, angle = augment(center_image, left_image, right_image, angle)

                image = preprocess(center_image)
                batch_images[i] = image
                batch_angles[i] = angle

            # yield sklearn.utils.shuffle(batch_images, batch_angles)
            yield batch_images, batch_angles


def batch_generator2(img_dir, X_data, y_data, batch_size=40, is_training=True):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    angles = np.empty(batch_size)

    while True:
        i = 0
        for idx in np.random.permutation(X_data.shape[0]):
            image_names = X_data[idx]
            angle = y_data[idx]
            center_image, left_image, right_image = load_images(img_dir, image_names)
            if is_training and np.random.rand() < 0.6:
                image, angle = augment(center_image, left_image, right_image, angle)

            image = preprocess(center_image)
            images[i] = image
            angles[i] = angle
            i += 1
            if i >= batch_size:
                break

        yield images, angles


def random_show_image(img_dir, X_data, y_data):
    import matplotlib.pyplot as plt

    idx = np.random.choice(X_data.shape[0])
    image_names = X_data[idx]
    angle = y_data[idx]

    center_image, left_image, right_image = load_images(img_dir, image_names)
    image, _ = augment(center_image, left_image, right_image, angle)
    image = preprocess(image)

    plt.imshow(image)
    plt.show()
