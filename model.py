"""
Improvement ideas:
- Use screenshots from several runs
- Challenge videos
- Recovery videos
- Reverse videos
- Gather smoother steering input (mouse, wheel?)
- Gauss smoothing of the road surface
- CLAHE
- HSL: emphasize colors
- nVidia algorithm
- Add throttle / brake
"""

import cfg
import reader
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPool2D, Activation, Dropout, Cropping2D
import matplotlib.pyplot as plt
import os
import data_manip


if cfg.GPU:
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def show_img(image):
    plt.imshow(image)
    plt.show()


def lenet_v1(x, y):
    model = Sequential()
    model.add(Cropping2D(((30, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(filters=16, kernel_size=(7,7), strides=(2,2), padding='same'))  # 160x320x16
    model.add(MaxPool2D((4,4)))  # 40x80x16
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(MaxPool2D((4,4)))  # 10*20*64
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(MaxPool2D((2,2)))  # 5*10*256
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(x, y, validation_split=0.2, shuffle=True, epochs=3)
    model.save(cfg.path_model)


def lenet_v2(x, y):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Conv2D(filters=16, kernel_size=(7,7), strides=(2,2), padding='same'))  # 160x320x16
    model.add(MaxPool2D((4,4)))  # 40x80x16
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(MaxPool2D((4,4)))  # 10*20*64
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(MaxPool2D((2,2)))  # 5*10*256
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(x, y, validation_split=0.2, shuffle=True, epochs=3)
    model.save(cfg.path_model)


def main(dataset, verbose=False):
    X_train, y_train = reader.read_sim_data(dataset)
    X_train, y_train = data_manip.preprocess(X_train, y_train)
    if verbose:
        data_manip.show_example(X_train, y_train, 0, 3, len(X_train) // 2)

    lenet_v1(X_train, y_train)


if __name__ == '__main__':
    main(dataset=1, verbose=False)

