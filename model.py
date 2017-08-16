#!/usr/bin/env python

import os

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, InputLayer, Dropout, MaxPool2D, Cropping2D, Lambda

import cv2


def build_model():
    image_shape = (160, 320, 3)
    model = Sequential()
    model.add(InputLayer(input_shape=image_shape))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Conv2D(18, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.75))

    model.add(Conv2D(6, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.75))

    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dropout(0.75))

    model.add(Dense(84))
    model.add(Dropout(0.75))
    model.add(Dense(1))

    return model


def train(model, x, y):
    model.compile(loss='mse', optimizer='adam')
    model.fit(x, y, epochs=10, batch_size=256, validation_split=0.2, shuffle=True)
    model.save('model.h5')


def load_driving_log(filename):
    prefix = os.path.split(filename)[0]
    log = pd.DataFrame.from_csv(filename, header=0, index_col=None)
    images = np.zeros([log.shape[0], 160, 320, 3], dtype=np.uint8)
    for i, path in enumerate(log['center']):
        image = cv2.imread(os.path.join(prefix, path))
        images[i, :, :, :] = image

    return images, np.array(log['steering'])


def main():
    model = build_model()
    x, y = load_driving_log('data/driving_log.csv')
    train(model, x, y)


if __name__ == '__main__':
    main()
