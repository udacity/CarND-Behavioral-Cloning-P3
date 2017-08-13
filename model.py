#!/usr/bin/env python
# encoding: utf-8

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def build_model(keep_prob):
    """
    Build NVIDIA model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Dropout(keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()

    return model


def compile_model(model, loss='mse', optimizer='adam'):
    model.compile(loss, optimizer)


def train_model(model, X_data, y_data, epochs):
    model.fit(X_data, y_data, validation_split=0.2, shuffle=True, nb_epoch=epochs)
    model.save('model.h5')


history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=len(train_samples),
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples),
                                     nb_epoch=5,
                                     verbose=1)


### print the keys contained in the history object
print(history_object.history.keys())


### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# 1. build model

# 2. fit model on training data

# 3. Evaluate
score = model.evaluate(X_test, Y_test, verbose=0)