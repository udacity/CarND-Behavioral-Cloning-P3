""""---------------------------------------
        Import statements
---------------------------------------"""

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
import helperFunctions as HF

'''---------------------------------------
        Hyper Parameters
---------------------------------------'''
correction = 0.25
epochs = 25
border_mode = 'same'
activation = 'elu'
optimizer = 'adam'
batch_size = 64

# Data Selection
myData      = True
uData       = True

# Data Visualization
preprocess  = True
getHist     = True
show        = True

# Pickle
usePickles = True
'''---------------------------------------
        Create Arrays
---------------------------------------'''

# Udacity Arrays
lines = []
images = []
measurements = []

# My Arrays
myLines = []
myImages = []
myMeasurements = []

# Combined Array
augmented_images = []
augmented_measurements = []

# Histogram Data
hist = []
bins = []
num_bins = 23

# Temp arrays for balancing Data-set
newImages = []
newMeasurements = []

'''---------------------------------------
        Data File Variables
---------------------------------------'''
myFile = './myData/driving_log.csv'
uFile = './data/driving_log.csv'

myImagePath = "./myData/IMG/"
uImagePath = "./data/IMG/"

mySplitToken = '\\'
uSplitToken = '/'

'''---------------------------------------
        Get Data
---------------------------------------'''

# Udacity Data
print("------------------------------------")
if uData:
    print("Using Ud Data")

    # Load Pickled Data
    if usePickles:
        images = np.load('./udacity_images.npy')
        measurements = np.load('./udacity_measurements.npy')
        print('loaded pickles')
    else:
        HF.read_data_from_file(uFile, lines)
        HF.get_images_and_measurements(lines, uSplitToken, uImagePath, images, measurements)
        np.save('./udacity_images', images)
        np.save('./udacity_measurements', measurements)

        # Flip Data
        HF.flip_image_and_measurement(images, measurements, augmented_images, augmented_measurements)

    # Histogram
    if getHist:
        HF.print_histogram(measurements, show=True, title="Udacity Data, before image augmentation")
        HF.print_histogram(augmented_measurements, show=True, title="Udacity Data, after image augmentation")



# My Data
if myData:
    print("Using My Data")

    # Load Pickled Data
    if usePickles:
        myImages = np.load('./my_images.npy')
        myMeasurements = np.load('./my_measurements.npy')

    # Load Data, save pickles
    else:
        HF.read_data_from_file(myFile, myLines)
        HF.get_images_and_measurements(myLines, mySplitToken, myImagePath, myImages, myMeasurements)
        np.save('./my_images.npy', myImages)
        np.save('./my_measurements.npy', myMeasurements)

        # Flip Data
        HF.flip_image_and_measurement(myImages, myMeasurements, augmented_images, augmented_measurements)

    # Histogram
    if getHist:
        HF.print_histogram(myMeasurements, show=True, title="My Data, before image augmentation")
        HF.print_histogram(augmented_measurements, show=True, title="My Data, after image augmentation")

# Combined Data
if usePickles:
    print("Loading Augmented Measurements and Images")
    augmented_measurements = np.load('./augmented_measurements.npy')
    augmented_images = np.load('./augmented_images.npy')
    print("Success")

else:
    '''---------------------------------------
            PreProcess Data Set
                - Convert RGB to YUV
                - Balance Data Set
                - Save Pickles
    ---------------------------------------'''
    hist, bins = np.histogram(augmented_measurements, num_bins)
    averageHeight = int(sum(hist) / num_bins)

    # Balance Data Set
    print("Balance Data Set")
    HF.balance_data_set(augmented_images, augmented_measurements, hist, bins, averageHeight, newImages, newMeasurements,
                        lowerLimit=-0.3, upperLimit=0.3)

    for image, measurement in zip(newImages, newMeasurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)

    print("Augmented Measurement = ", len(augmented_measurements))

    # Check if Data Set is Balanced
    HF.print_histogram(augmented_measurements, show=True, title="Combined Data, after data balance")

    # Save Augmented Array Pickles
    np.save('./augmented_measurements', augmented_measurements)
    np.save('./augmented_images', augmented_images)
    print("Augmented Pickles Saved")

if preprocess:
    print("Preprocess = ", preprocess)
    for image in augmented_images:
        image = HF.preprocess_image(image)

'''---------------------------------------
        Sanity Checks
---------------------------------------'''

print("\n\n==============================================\n")
print("Ud Data\n---------------------------------")
print("Lines -- ", len(lines))
print("3x    -- ", len(lines)*3)
print("Images-- ", len(images))
print("Measur-- ", len(measurements))

print("My Data\n---------------------------------")
print("Lines -- ", len(myLines))
print("3x    -- ", len(myLines)*3)
print("Images-- ", len(myImages))
print("Measur-- ", len(myMeasurements))

print("Totals\n---------------------------------")
combined = len(myMeasurements) + len(measurements)
print("Combined        -- ", combined)
print("2x Combined     -- ", combined * 2)
print("Augmented Images-- ", len(augmented_images))
print("Augmented Measur-- ", len(augmented_measurements))

# if (combined * 2) != len(augmented_measurements):
#     exit(2)
# if (combined * 2) != len(augmented_images):
#     exit(2)

print("\n(2)Combined = Aug_Meas == Aug_Images == ", combined*2)
print("==============================================\n\n")



'''---------------------------------------
        Declare TF Variables
---------------------------------------'''
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

'''---------------------------------------
        Define NN Model
---------------------------------------'''
model = Sequential()

# Preprocess
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# Convolutions
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation=activation, border_mode=border_mode))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation=activation, border_mode=border_mode))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation=activation, border_mode=border_mode))

model.add(Convolution2D(64, 3, 3, activation=activation, border_mode=border_mode))
model.add(Convolution2D(64, 3, 3, activation=activation, border_mode=border_mode))

# Fully Connected
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

'''---------------------------------------
        CheckPoint
---------------------------------------'''
checkpoint = ModelCheckpoint('model-rgb-{epoch:02d}.h5',
                             monitor='val_loss',
                             verbose=2,
                             save_best_only=False,
                             mode='auto')
'''---------------------------------------
        Build and Run
---------------------------------------'''
model.compile(optimizer=optimizer, loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True,
          verbose=2, nb_epoch=epochs, batch_size=batch_size, callbacks=[checkpoint])


'''---------------------------------------
        Save and Exit
---------------------------------------'''
model.save('model.h5')
print("Model Saved")
exit()
