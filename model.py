#!/usr/bin/env python
# encoding: utf-8

import os
import math
import argparse
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
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


def compile_model(model, learning_rate):
    """
    Compile the model
    """
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))


def train_model(model, train_gen, n_train, validation_gen, n_validation, n_epochs):
    """
    Train the model
    """
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    history = model.fit_generator(generator=train_gen,
                                  samples_per_epoch=n_train,
                                  nb_epoch=n_epochs,
                                  validation_data=validation_gen,
                                  nb_val_samples=n_validation,
                                  callbacks=[checkpoint],
                                  verbose=1)
    model.save('model.h5')
    return history


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
    # parse command-line
    parser = argparse.ArgumentParser(description='CarND Behavioral Cloning')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    # split validation set from training set
    X, y = utils.load_csv(args.data_dir)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    # create train and validation generator
    img_dir = os.path.join(args.data_dir, 'IMG')
    train_generator = utils.batch_generator(img_dir, X_train, y_train, batch_size=args.batch_size)
    validation_generator = utils.batch_generator(img_dir, X_valid, y_valid, batch_size=args.batch_size, is_training=False)

    # build model
    model = build_model(args.keep_prob)

    # compile the model
    compile_model(model, args.learning_rate)

    # train the model
    num_train_samples = math.ceil(len(X_train) / args.batch_size) * args.batch_size
    history = train_model(model,
                          train_generator,
                          num_train_samples,
                          validation_generator,
                          len(X_valid),
                          args.nb_epoch)

    draw_metrics(history)


if __name__ == "__main__":
    main()
