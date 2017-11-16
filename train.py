import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda
import sys
import argparse
import os
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import json
from keras.callbacks import TensorBoard
from time import time



def parse_args(arguments):
    """
    Parses arguments given at the command line.
    :param arguments: Arguments given at the command line
    :return: Dict of variables parsed from the arguments
    """
    parser = argparse.ArgumentParser(description="Trains a behavioral cloning model from a given training file set.")
    parser.add_argument('-c', '--configuration', help="File path for optional configuration file.",
                        dest='config')

    return vars(parser.parse_args(arguments))


def load_config(config_name):
    """
    loads a json config file and returns a config dictionary
    """
    with open(config_name) as config_file:
        configuration = json.load(config_file)
        return configuration

def get_log_lines(path):
    """
    Gets list of records from driving log.
    :param path: Input path for driving log.
    :return: List of driving log records.
    """
    lines = []
    with open(path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def get_images_and_measurements(path, lines):
    """
    Gets images and measurements from training images.
    :param path: Input path for images.
    :return:
    """
    images = []
    measurements = []

    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = path + '/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)

        measurement = float(line[3])
        measurements.append(measurement)

    return np.array(images), np.array(measurements)

def create_model(units=1, loss_function='mse', optimizer='adam', input_shape=(160,320,3)):
    """
    Constructs Keras model object
    :return: Compiled Keras model object
    """
    model = Sequential()
    #model.add(Convolution2D(160, 3, 3, input_shape=input_shape))
    #model.add(MaxPooling2D((2,2)))
    #model.add(Dropout(0.5))
    #model.add(Activation('relu'))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(units))

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    return model

def custom_get_params(self, **params):
    """
    Function to patch issue in Keras
    :param self:
    :param params:
    :return:
    """
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res



if __name__ == '__main__':

    # Set TensorFlow logging so it isn't so verbose.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Load the necessary parameters
    args = parse_args(sys.argv[1:])
    config = load_config(args['config'])

    # Load data
    lines = get_log_lines(config['input_path'])
    images, measurements = get_images_and_measurements(config['input_path'], lines)
    use_grid_search = config['use_grid_search']

    # Designate X and y data
    X_train = images
    y_train = measurements

    if config["use_grid_search"] == 'True':
        from keras.wrappers.scikit_learn import BaseWrapper
        import copy

        # Fix the Keras deep copy problem
        BaseWrapper.get_params = custom_get_params

        # Set the model
        model = KerasRegressor(build_fn=create_model, verbose=1)

        # Set up the parameter grid
        # TODO: Check to see which config items that aren't used for something else are lists.
        epochs = config["epochs"]

        param_grid = dict(epochs=epochs)
        grid = GridSearchCV(estimator=model, param_grid=param_grid)

        # Establish tensorboard
        tensorboard = TensorBoard(log_dir=config["tensorboard_log_dir"] + "/{}".format(time()))

        # Fit the model using Grid Search
        grid_result = grid.fit(X_train, y_train, fit_params={'callbacks': [tensorboard], 'shuffle':True, 'validation_split':0.5})

        # Display model output
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stdvs = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, parameters in zip(means, stdvs, params):
            print("%f (%f) with: %r" % (mean, stdev, parameters))

    else:
        model = create_model(config['units'])

        # Establish tensorboard
        if config["use_tensorboard"] == "True":
            tensorboard = TensorBoard(log_dir=config["tensorboard_log_dir"] + "/{}".format(time()), histogram_freq=1)
            model.fit(X_train, y_train, nb_epoch=config['epochs'],
                  validation_split=0.5, shuffle=True, callbacks=[tensorboard])
        else:
            model.fit(X_train, y_train, nb_epoch=config['epochs'],
                  validation_split=0.5, shuffle=True)


        if config['output_path'].endswith('.h5'):
            model.save(config['output_path'])
        else:
            model.save(config['output_path'] + '.h5')

    sys.exit(0)