import csv
from cfg import *
import os
import cv2
import numpy as np


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
        source_path = line[0]
        filename = os.path.basename(source_path)
        current_path = path_current_root + CFG.path_img + filename
        if os.path.exists(current_path):
            image = cv2.imread(current_path)
            image = image[...,::-1]
            images.append(image)
            steering_measurement = float(line[3])
            steering_measurements.append(steering_measurement)
            success += 1
        else:
            print('WARNING! File missing: {}'.format(current_path))
            pass
    images = np.array(images)
    steering_measurements = np.array(steering_measurements)
    print('From a total of {} successfully read {} images.'.format(len(csv_lines), success))
    return images, steering_measurements


if __name__ == '__main__':
    read_sim_data(1)
