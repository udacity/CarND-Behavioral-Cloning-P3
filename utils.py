p#!/usr/bin/env python
# encoding: utf-8

import os
import csv
import cv2
import random
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split


CORRECTION = 0.2
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def read_csv(file):
    samples = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples


def load_csv(file):
    """
    Load training data 
    """
    data_df = pd.read_csv(file)
    samples = data_df[['center', 'left', 'right', 'steering']].values
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
    angles = (angle, angle-CORRECTION, angle+CORRECTION)

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
    angle = angle+trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))

    return image, angle


def augment(center_image, left_image, right_image, angle):
    """
    Augment images through flip, shift and brightness tuning
    """
    angles = [angle, angle-CORRECTION, angle+CORRECTION]

    # 1. randomly pick up a image
    image, angle = pick_image(center_image, left_image, right_image, angle)

    # 2. add flip image in order to recognize both clockwise and counter-clockwise roads
    image, angle = random_flip(image, angle) 

    # 3. randomly adjust shift
    images, angles = random_translation(image, angle)

    # 4. randomly adjust brightness
    image = random_brightness(image)

    return image, angle


def load_images(img_dir, img_names):
    return [cv2.imread(os.path.join(img_dir, img_names[i].split('/')[-1])) for i in range(3)]


def generator(samples, img_dir, batch_size=40, is_training=True):
    num_samples = len(samples)
    while True # Loop forever so the generator never terminates
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


def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def batch_generator1(img_dir, X_data, y_data, batch_size=32, is_training=True):

    num_of_batches = np.ceil(X_data.shape[0]/batch_size)
    cnt = 0
    indices = np.arange(X_data.shape[0])
    np.random.shuffle(indices)

    while True # Loop forever so the generator never terminates

        i = 0
        for idx in np.random.permutation(X_data.shape[0]):
            center_name, left_name, right_name = X_data[idx]
            angle = y_data[idx]
            center_image, left_image, right_image = load_images(img_dir, image_names)
            if is_training and np.random.rand() < 0.6:
                image, angle = augment(center_image, left_image, right_image, angle)
            
            image = preprocess(image)
            images[i] = image
            angles[i] = angle   
            i += 1
            if i == batch_size:
                break
        yield images, angles


def batch_generator2(img_dir, X_data, y_data, batch_size=32, is_training=True):

    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    angles = np.empty(batch_size)

    while True # Loop forever so the generator never terminates
        i = 0
        for idx in np.random.permutation(X_data.shape[0]):
            center_name, left_name, right_name = X_data[idx]
            angle = y_data[idx]
            center_image, left_image, right_image = load_images(img_dir, image_names)
            if is_training and np.random.rand() < 0.6:
                image, angle = augment(center_image, left_image, right_image, angle)
            
            image = preprocess(image)
            images[i] = image
            angles[i] = angle   
            i += 1
            if i >= batch_size:
                break
        yield images, angles
        
