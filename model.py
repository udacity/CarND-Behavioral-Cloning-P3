#!/usr/bin/env python
# encoding: utf-8

import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout
from sklearn.model_selection import train_test_split
import utils


def build_model(keep_prob):
    """
    Build NVIDIA model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=utils.INPUT_SHAPE))
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
    """
    Compile the model
    """
    model.compile(loss, optimizer)


def train_model(model, train_gen, n_train, validation_gen, n_validation, n_epochs):
    """
    Train the model
    """
    history = model.fit_generator(generator=train_gen,
                                  steps_per_epoch=n_train,
                                  validation_data=validation_gen,
                                  validation_steps=n_validation,
                                  epochs=n_epochs,
                                  verbose=1)
    model.save('model.h5')
    return history


def get_history_keys(history_object):
    # print the keys contained in the history object
    return history_object.history.keys()


def draw_metrics(history_object):
    """
    plot the training and validation loss for each epoch
    """
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def main():
    csv_file = 'data/xxx.csv'
    img_dir = 'data/IMG/'
    epochs = 5
    keep_prob = 0.5

    # split validation set from training set
    train_samples, validation_samples = train_test_split(utils.read_csv(csv_file), test_size=0.33)

    # create train and validation generator    
    train_generator = utils.generator(train_samples, img_dir)
    validation_generator = utils.generator(validation_samples, img_dir)

    # build model
    model = build_model(keep_prob)

    # compile the model
    compile_model(model)

    # train the model
    history = train_model(model, 
                          train_generator, 
                          len(train_samples), 
                          validation_generator, 
                          len(validation_samples), 
                          epochs)
        
    # plot loss
    draw_metrics(history)

    # print history
    print(get_history_keys(history))
    
     

if __name__ == "__main__":
    main()

