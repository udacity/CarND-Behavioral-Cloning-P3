from cfg import *
from reader import read_sim_data
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense


def main(index):
    X_train, y_train = read_sim_data(index)
    print(X_train.shape)
    print(y_train.shape)

    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)
    model.save(CFG.path_model)


if __name__ == '__main__':
    main(1)

