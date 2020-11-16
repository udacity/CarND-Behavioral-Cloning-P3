from cfg import *
from reader import read_sim_data
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
import matplotlib.pyplot as plt
import os

GPU = False # or True
if GPU:
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def show_img(image):
    plt.imshow(image)
    plt.show()


def main(index):
    X_train, y_train = read_sim_data(index)
    # show_img(X_train[0])

    print(X_train.shape)
    print(y_train.shape)

    model = Sequential()
    model.add(Lambda(lambda x: x - 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=15)
    model.save(CFG.path_model)


if __name__ == '__main__':
    main(1)

