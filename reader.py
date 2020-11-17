import csv
import cfg
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_line(data_idx, csv_line, side='center', verbose=True):
    column = 0
    measurement = float(csv_line[3])
    if side == 'left':
        column = 1
        measurement += cfg.camera_offset_steer
    elif side == 'right':
        column = 2
        measurement -= cfg.camera_offset_steer

    path = cfg.path_fmt_data_root.format(data_idx) + cfg.path_img + os.path.basename(csv_line[column])

    if os.path.exists(path):
        image = cv2.imread(path)
        images = image[..., ::-1]
        return image, measurement
    else:
        if verbose:
            print('WARNING! File missing: {}'.format(path))
        return None, None




def read_sim_data(dataset=1, verbose=False):
    # read csv
    path_current_root = cfg.path_fmt_data_root.format(dataset)
    path = path_current_root + cfg.path_log
    print('Reading file {}...'.format(path))
    csv_lines = []
    with open(path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            csv_lines.append(line)
    print('File read successfully. First line:\n', csv_lines[0])

    # read images
    print('Reading images...')
    images = []
    steering_measurements = []
    success = 0
    for line in csv_lines:
        image, steering_measurement = read_line(dataset, line, side='center', verbose=verbose)
        if image is not None and steering_measurement is not None:
            images.append(image[...,::-1])
            steering_measurements.append(steering_measurement)
            success += 1
        image, steering_measurement = read_line(dataset, line, side='left', verbose=verbose)
        if image is not None and steering_measurement is not None:
            images.append(image[...,::-1])
            steering_measurements.append(steering_measurement)
            success += 1
        image, steering_measurement = read_line(dataset, line, side='right', verbose=verbose)
        if image is not None and steering_measurement is not None:
            images.append(image[...,::-1])
            steering_measurements.append(steering_measurement)
            success += 1
    images = np.array(images)
    steering_measurements = np.array(steering_measurements)
    print('From a total of {} successfully read {} images to dataset.'.format(len(csv_lines * 3), success))
    return images, steering_measurements


def read_datasets(first, last, verbose=False):
    X_train, y_train = read_sim_data(first, verbose)
    if last > first:
        for i in range(first + 1, last + 1):
            X, y = read_sim_data(i, verbose)
            X_train = np.vstack((X_train, X))
            y_train = np.hstack((y_train, y))
    if verbose:
        print('Loaded {} images from datasets {}..{} (both inclusive).'.format(X_train.shape[0], first, last))
    return X_train, y_train


if __name__ == '__main__':
    read_datasets(first=1, last=3, verbose=True)
