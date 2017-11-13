'''---------------------------------------
        Import Statements
---------------------------------------'''

import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import random
from tempfile import TemporaryFile


correction = 0.25
num_bins = 23
colorConversion = cv2.COLOR_BGR2YUV

'''---------------------------------------
        Read data from File
---------------------------------------'''
def read_data_from_file(fileName, lineArray):
    with open(fileName) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lineArray.append(line)

'''---------------------------------------
        Extract images and Measurements
---------------------------------------'''


def get_images_and_measurements(lineArray, splitToken, imagePath, imageArray, measurementArray):
    for line in lineArray:
        for i in range(3):
            source_path = line[i]
            tokens = source_path.split(splitToken)
            filename = tokens[-1]
            local_path = imagePath + filename
            image = cv2.imread(local_path)
            imageArray.append(image)
        measurement = float(line[3])
        measurementArray.append(measurement)
        measurementArray.append(measurement + correction)
        measurementArray.append(measurement - correction)

'''---------------------------------------
        Print Histogram of Data
---------------------------------------'''
def print_histogram(measurement_array, show, title = ''):
    avg_samples_per_bin = len(measurement_array)/num_bins
    hist, bins = np.histogram(measurement_array, num_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.plot((np.min(measurement_array), np.max(measurement_array)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
    if show:
        plt.title(title)
        plt.show()

'''---------------------------------------
        Flip each image and measurement
---------------------------------------'''


def flip_image_and_measurement(imageArray, measurementArray, augmented_images, augmented_measurements):
    for image, measurement in zip(imageArray, measurementArray):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        flipped_image = cv2.flip(image, 1)
        flipped_measurement = measurement * -1.0
        augmented_images.append(flipped_image)
        augmented_measurements.append(flipped_measurement)


'''---------------------------------------
        PreProcess Image
---------------------------------------'''
def preprocess_image(img):
    new_img = cv2.cvtColor(img, colorConversion)
    return new_img

'''---------------------------------------
        Balance DataSet
---------------------------------------'''
def balance_data_set(augmented_images, augmented_measurements, hist, bins, averageHeight, newImages, newMeasurements, lowerLimit, upperLimit):
    for image, measurement in zip(augmented_images, augmented_measurements):
        if (measurement < lowerLimit or measurement > upperLimit):
            for i in range(num_bins):
                if bins[i] < measurement < bins[i + 1]:
                    print(bins[i], " < ", measurement, " < ", bins[i + 1])
                    difference = abs(averageHeight - hist[i])
                    multiples = int(difference / hist[i])
                    for k in range(multiples):
                        brightness = random.randint(0, 100)
                        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                        y, u, v = cv2.split(yuv)
                        y -= brightness
                        final_yuv = cv2.merge((y, u, v))
                        newImage = cv2.cvtColor(final_yuv, cv2.COLOR_YUV2BGR)
                        newImages.append(newImage)
                        newMeasurements.append(measurement)