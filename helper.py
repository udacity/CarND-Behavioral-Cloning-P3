import errno
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc
from scipy.ndimage import rotate
from scipy.stats import bernoulli

# Some useful constants
DRIVING_LOG_FILE = './data/driving_log.csv'
IMG_PATH = './data/'
STEERING_COEFFICIENT = 0.229


def randomly_drop_low_steering_data(data):
    index = [i for i in range(len(data)) if abs(data[i][1])<0.05]
    unwanted = [data[i] for i in index if np.random.randint(10) < 8]
    data = [e for e in data if e not in unwanted]
    return data
    
def randomly_drop_extremely_high_steering_data(data):
    index = [i for i in range(len(data)) if abs(data[i][1])>10.5]
    unwanted = [data[i] for i in index if np.random.randint(10) < 8]
    data = [e for e in data if e not in unwanted]
    return data
    

def crop(image, top_percent, bottom_percent):
    return image[60:135, : ]

def resize(image, new_dim):
    return scipy.misc.imresize(image, new_dim)


def random_flip(image, steering_angle, flipping_prob=0.5):
    head = bernoulli.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle


def random_gamma(image):
    """
    Random gamma correction is used as an alternative method changing the brightness of
    training images.
    http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    """
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def random_shear(image, steering_angle, shear_range=200):
    """
    Source: https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
    """
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return image, steering_angle


def generate_new_image(image, steering_angle, top_crop_percent=0.35, bottom_crop_percent=0.1,
                       resize_dim=(64, 64), do_shear_prob=0.9):
    head = bernoulli.rvs(do_shear_prob)
    if head == 1:
        image, steering_angle = random_shear(image, steering_angle)

    image = crop(image, top_crop_percent, bottom_crop_percent)
    image, steering_angle = random_flip(image, steering_angle)
    image = random_gamma(image)
    image = resize(image, resize_dim)
    
    return image, steering_angle


def get_next_image_files(batch_size=64):
    data = pd.read_csv(DRIVING_LOG_FILE)
    num_of_img = len(data)
    rnd_indices = np.random.randint(0, num_of_img, batch_size)

    image_files_and_angles = []
    for index in rnd_indices:
        rnd_image = np.random.randint(0, 3)
        if rnd_image == 0:
            img = data.iloc[index]['left'].strip()
            angle = data.iloc[index]['steering'] + STEERING_COEFFICIENT
            image_files_and_angles.append((img, angle))

        elif rnd_image == 1:
            img = data.iloc[index]['center'].strip()
            angle = data.iloc[index]['steering']
            image_files_and_angles.append((img, angle))
        else:
            img = data.iloc[index]['right'].strip()
            angle = data.iloc[index]['steering'] - STEERING_COEFFICIENT
            image_files_and_angles.append((img, angle))

    #image_files_and_angles = randomly_drop_low_steering_data(image_files_and_angles)
    #image_files_and_angles = randomly_drop_extremely_high_steering_data(image_files_and_angles)

    return image_files_and_angles


def generate_next_batch(batch_size=64):
    while True:
        X_batch = []
        y_batch = []
        images = get_next_image_files(batch_size)
        for img_file, angle in images:
            raw_image = plt.imread(IMG_PATH + img_file)
            raw_angle = angle
            new_image, new_angle = generate_new_image(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)

        
        yield np.array(X_batch), np.array(y_batch)


def save_model(model, model_name='model.h5', json_name='model.json',
               weights_name='model_wts.h5'):
    """
    Save the model into the hard disk
    """
    json_string = model.to_json()
    with open(json_name, 'w') as outfile:
        json.dump(json_string, outfile)

    model.save_weights(weights_name)
    
    model.save(model_name)


