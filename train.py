import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout
import sys
import argparse
import os
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import json
from keras.wrappers.scikit_learn import BaseWrapper
import copy


def parse_args(arguments):
    """
    Parses arguments given at the command line.
    :param arguments: Arguments given at the command line
    :return: Dict of variables parsed from the arguments
    """
    parser = argparse.ArgumentParser(description="Trains a behavioral cloning model from a given training file set.")
    parser.add_argument('-i', '--input', help="Input run name starter stem. Will be appropriately split to open the driving log and the image directory.",
                        dest='input_path')
    parser.add_argument('-o', '--output', help="Output destination to save model.", dest='output_path')
    parser.add_argument('-l', '--loss', choices=['mse','mean_absolute_error',
                                                 'mean_squared_logarithmic_error',
                                                 'squared_hinge','hinge','categorical_hinge',
                                                 'logcosh','categorical_crossentropy',
                                                 'sparse_categorical_crossentropy',
                                                 'binary_crossentropy',
                                                 'kullback_leibler_divergence',
                                                 'poisson','cosine_proximity'], help="Loss function type to use.",
                        default='mse', dest='loss_function')
    parser.add_argument('-p', '--optimizer', choices=['adam','SGD','RMSprop',
                                                      'Adagrad','Adadelta','Adamax',
                                                      'Nadam','TFOptimizer'], default='adam',
                        dest='optimizer')
    parser.add_argument('-e', '--epochs', help="Number of epochs to train the model for.",
                        default=40, dest='num_epochs')
    parser.add_argument('-v', '--validation-split', help="Percentage of data to hold out for validation.",
                        default=0.2, dest='validation_split')
    parser.add_argument('-s', '--shuffle', help="Shuffles data prior to training.",
                        choices=['True', 'False'], default='True', dest='shuffle')
    parser.add_argument('-u', '--units', help="Number of dense units per layer.",
                        default=1, dest='units')
    parser.add_argument('-c', '--configuration', help="File path for optional configuration file.",
                        dest='config')
    parser.add_argument('-g', '--grid-search', help="Indicates whether or not to use Grid Search in training.",
                        choices=['True', 'False'], default='False', dest='use_grid_search')
    parser.add_argument('-f', '--folds', help='Number of folds to use for k-crossfolds validation.',
                        dest='folds')
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

def create_model(units=1, loss_function='mse', optimizer='adam', epochs=10, input_shape=(160,320,3)):
    """
    Constructs Keras model object
    :return: Compiled Keras model object
    """
    model = Sequential()
    model.add(Convolution2D(160, 3, 3, input_shape=input_shape))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
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

    BaseWrapper.get_params = custom_get_params

    # Set TensorFlow logging so it isn't so verbose.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = parse_args(sys.argv[1:])
    if args['config']:
        config = load_config(args['config'])
        lines = get_log_lines(config['input_path'])
        images, measurements = get_images_and_measurements(config['input_path'], lines)
        use_grid_search = config['use_grid_search']

        X_train = images
        y_train = measurements

        if use_grid_search == 'True':
            model = KerasRegressor(build_fn=create_model, verbose=0)

            epochs = config["epochs"]

            param_grid = dict(epochs=epochs)

            grid = GridSearchCV(estimator=model, param_grid=param_grid)

            grid_result = grid.fit(X_train, y_train)

            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, parameters in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, parameters))
        else:
            model = create_model(config['units'][0])
            model.fit(X_train, y_train, nb_epoch=config['epochs'][0],
                      validation_split=0.5, shuffle=True)
            if config['output_path'].endswith('.h5'):
                model.save(config['output_path'])
            else:
                model.save(config['output_path'] + '.h5')


    else:
        lines = get_log_lines(args['input_path'])

        images, measurements = get_images_and_measurements(args['input_path'], lines)
        use_grid_search = args['use_grid_search']

        X_train = images
        y_train = measurements

        model = create_model(args['units'])
        model.fit(X_train, y_train, nb_epoch=args['num_epochs'],
                  validation_split=args['validation_split'], shuffle=args['shuffle'])

        if args['output_path'].endswith('.h5'):
            model.save(args['output_path'])
        else:
            model.save(args['output_path'] + '.h5')

    sys.exit(0)