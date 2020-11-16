import csv
from cfg import *
import os
import cv2
import numpy as np


def read_line(data_idx, csv_line, side='center'):
    column = 0
    measurement = float(csv_line[3])
    if side == 'left':
        column = 1
        measurement += CFG.camera_offset_steer
    elif side == 'right':
        column = 2
        measurement -= CFG.camera_offset_steer

    path = CFG.path_fmt_data_root.format(data_idx) + CFG.path_img + os.path.basename(csv_line[column])

    if os.path.exists(path):
        image = cv2.imread(path)
        images = image[..., ::-1]
        return image, measurement
    else:
        print('WARNING! File missing: {}'.format(path))
        return None, None


def read_sim_data(index=1):
    # read csv
    path_current_root = CFG.path_fmt_data_root.format(index)
    path = path_current_root + CFG.path_log
    print('Reading file {}...'.format(path))
    csv_lines = []
    with open(path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            csv_lines.append(line)
    print('File read successfully.', csv_lines[0])

    # read images
    print('Reading images...')
    images = []
    steering_measurements = []
    success = 0
    for line in csv_lines:
        image, steering_measurement = read_line(index, line, side='center')
        if image is not None and steering_measurement is not None:
            images.append(image)
            steering_measurements.append(steering_measurement)
            success += 1
        image, steering_measurement = read_line(index, line, side='left')
        if image is not None and steering_measurement is not None:
            images.append(image)
            steering_measurements.append(steering_measurement)
            success += 1
        image, steering_measurement = read_line(index, line, side='right')
        if image is not None and steering_measurement is not None:
            images.append(image)
            steering_measurements.append(steering_measurement)
            success += 1

    images = np.array(images)
    steering_measurements = np.array(steering_measurements)
    print('From a total of {} successfully read {} images.'.format(len(csv_lines * 3), success))
    return images, steering_measurements


if __name__ == '__main__':
    read_sim_data(1)
