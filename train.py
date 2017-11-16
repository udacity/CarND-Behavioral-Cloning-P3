import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
import sys
import argparse
import os


def parse_args(arguments):
    """
    Parses arguments given at the command line.
    :param arguments: Arguments given at the command line
    :return: Dict of variables parsed from the arguments
    """
    parser = argparse.ArgumentParser(description="Trains a behavioral cloning model from a given training file set.")
    parser.add_argument('-i', '--input', help="Input run name starter stem. Will be appropriately split to open the driving log and the image directory.",
                        required=True, dest='input_path')
    parser.add_argument('-o', '--output', help="Output destination to save model.", required=True, dest='output_path')
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
                        choices=[True, False], default=True, dest='shuffle')

    return vars(parser.parse_args(arguments))

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

def create_model(input_shape=(160,320,3)):
    """
    Constructs Keras model object
    :return: Keras model object
    """
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1))

    return model


if __name__ == '__main__':

    # Set TensorFlow logging so it isn't so verbose.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = parse_args(sys.argv[1:])

    lines = get_log_lines(args['input_path'])

    images, measurements = get_images_and_measurements(args['input_path'], lines)

    X_train = images
    y_train = measurements

    model = create_model()

    model.compile(loss=args['loss_function'], optimizer=args['optimizer'])
    model.fit(X_train, y_train, nb_epoch=args['num_epochs'],
              validation_split=args['validation_split'], shuffle=args['shuffle'])

    if args['output_path'].endswith('.h5'):
        model.save(args['output_path'])
    else:
        model.save(args['output_path'] + '.h5')

    sys.exit(0)