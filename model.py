## IMPORTING LIBRARIES ##
#########################
import os
import csv
import cv2
import numpy as np

from PIL import Image
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout

from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K

## READING IMAGE DATA FROM THE LOG ##
#####################################
lines = []
header = True
camera_images = []
steering_angles = []

with open('data/driving_log.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if header:
            header = False
            continue
        steering_center = float(row[3])

        #Steering angle (sa) correction factor for stereo cameras
        sa_cor = 0.2
        steering_left = steering_center + sa_cor
        steering_right = steering_center - sa_cor

        #Reading camera images from their paths
        path_src1 = row[0]
        path_src2 = row[1]
        path_src3 = row[2]
        img_name1 = path_src1.split('/')[-1]
        img_name2 = path_src2.split('/')[-1]
        img_name3 = path_src3.split('/')[-1]
        path1 = 'data/IMG/' + img_name1 
        path2 = 'data/IMG/' + img_name2 
        path3 = 'data/IMG/' + img_name3 

        #Image and Steering Dataset
        img_center = np.asarray(Image.open(path1))
        img_left = np.asarray(Image.open(path2))
        img_right = np.asarray(Image.open(path3))
        camera_images.extend([img_center, img_left, img_right])
        steering_angles.extend([steering_center, steering_left, steering_right])

## DATA AUGMENTATION ##
#######################
augmented_imgs, augmented_sas= [],[]

for aug_img,aug_sa in zip(camera_images,steering_angles):
    augmented_imgs.append(aug_img)
    augmented_sas.append(aug_sa)
    
    #Flipping the image
    augmented_imgs.append(cv2.flip(aug_img,1))
    
    #Reversing the steering angle
    augmented_sas.append(aug_sa*-1.0)
  
## INDEPENDENT VARIABLES and LABELS ##
######################################
X_train, y_train = np.array(augmented_imgs), np.array(augmented_sas)
X_train, y_train = np.array(camera_images), np.array(steering_angles)

## IMAGE PRE-PROCESSING ##
##########################
def preprocess(image):
    import tensorflow as tf
    #Resizing the image
    return tf.image.resize_images(image, (200, 66))

## THE CNN ARCHITECTURE ##
##########################
'''
The CNN architecture is used from NVIDIA's End to End Learning for Self-Driving Cars paper.
Reference: https://arxiv.org/pdf/1604.07316v1.pdf
'''
#Keras Sequential Model
model = Sequential()

#Image cropping to get rid of the irrelevant parts of the image (the hood and the sky)
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

#Pre-Processing the image
model.add(Lambda(preprocess))
model.add(Lambda(lambda x: (x/ 127.0 - 1.0)))

#The layers
model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1))
print(model.summary())

#Compiling and Saving the Model
model.compile(loss='mse',optimizer='adam') #adaptive moment estimation.
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=10)
model.save('model.h5') 

print('The model.h5 file has been created!') 
## END OF THE CODE ##