#!/usr/bin/env python
# encoding: utf-8

import os
import csv
import cv2
import random
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


CORRECTION = 0.2
IMG_DIR = 'data/IMG/'
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def read_csv(file):
    samples = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[75:-25, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(images):
    """
    Combine all preprocess functions into one
    """
    new_images = []
    for image in images: 
        image = crop(image)
        image = resize(image)
        image = rgb2yuv(image)
        new_images.append(image)
    return new_images

def random_flip(image, angle):
    """
    Randomly flip the image to left or right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        angle = -angle
    return image, angle


def random_brightness(images, ratio=0.5):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    new_images = []
    for image in images:
        hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
        brightness = np.float64(hsv[:, :, 2])
        brightness = brightness * (1.0 + np.random.uniform(-ratio, ratio))
        brightness[brightness>255] = 255
        brightness[brightness<0] = 0
        hsv[:, :, 2] = brightness
        new_images.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    return new_images


def random_translation(images, angles, range_x=100, range_y=10):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    new_images, new_angles = [], []
    for i, image in enumerate(images):
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        new_angles.append(angles[i]+trans_x * 0.002)
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        new_images.append(cv2.warpAffine(image, trans_m, (width, height)))
    return new_images, new_angles


def flip_images(images, angles):
    return [cv2.flip(i, 1) for i in images], [a*(-1) for a in angles]


def augment(images, angles):
    """ 
    Augment images through flip, shift and brightness tuning
    """
    # 1. add flip images in order to recognize both clockwise and counter-clockwise roads
    flip_images, flip_angles = flip_images(images, angles)
    images.extend(flip_images)
    angles.extend(flip_angles)

    # 2. randomly adjust shift
    images, angles = random_translation(images, angles)

    # 3. randomly adjust brightness
    images = random_brightness(images) 

    return images, angles
    

def load_images(img_dir, sample):
    return [cv2.imread(os.path.join(img_dir, sample[i].split('/')[-1])) for i in range(3)]


def generator(samples, img_dir, batch_size=32, is_training=True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            batch_images = []
            batch_angles = []
            for batch_sample in batch_samples:
                angle = float(batch_sample[3])
                # angles list contains center_angle, left_angle and right_angle
                angles = [angle, angle-CORRECTION, angle+CORRECTION]
                # images list contains center_image, left_image and right_image
                images = load_images(img_dir, batch_sample)
                if is_training and np.random.rand() < 0.6:
                    images, angles = augment(images, angles)

                images = preprocess(images)
                batch_images.extend(images)
                batch_angles.extend(angles)
            X_data = np.array(batch_images)
            y_data = np.array(batch_angles)
            yield sklearn.utils.shuffle(X_data, y_data)


train_samples, validation_samples = train_test_split(samples, test_size=0.33)


# compile and train the model using the generator function
train_generator = generator(train_samples, IMG_DIR)
validation_generator = generator(validation_samples, IMG_DIR)