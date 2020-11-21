import csv
import cfg
import os
import cv2
import numpy as np
import sklearn.utils


def get_all_meta(first_dataset, last_dataset, check_files_exist=True, verbose=False):
    """
    Reads in all image paths and steering values from given folders (last exclusive).
    Schema: steering (float), img path (str).
    """
    meta_db = []
    for data_set in range(first_dataset, last_dataset):
        path = cfg.path_fmt_data_root.format(data_set) + cfg.path_log
        print('Reading file {}...'.format(path), end='')
        with open(path) as csv_file:
            reader = csv.reader(csv_file)
            count_possible, count_ok = 0, 0
            for line in reader:
                count_possible += 3
                if (not check_files_exist) or (os.path.exists(line[0])):  # center image
                    current_line_data = []
                    current_line_data.append(line[0])
                    current_line_data.append(float(line[3]))
                    meta_db.append([*current_line_data])
                    count_ok += 1
                else:
                    if verbose:
                        print('File missing:', line[0])
                if (not check_files_exist) or (os.path.exists(line[1])):  # left image
                    current_line_data = []
                    current_line_data.append(line[1])
                    current_line_data.append(float(line[3]))
                    meta_db.append([*current_line_data])
                    count_ok += 1
                else:
                    if verbose:
                        print('File missing:', line[1])
                if (not check_files_exist) or (os.path.exists(line[2])):  # right image
                    current_line_data = []
                    current_line_data.append(line[2])
                    current_line_data.append(float(line[3]))
                    meta_db.append([*current_line_data])
                    count_ok += 1
                else:
                    if verbose:
                        print('File missing:', line[2])
            print('done. From {} read {} items successfully.{}'.format(count_possible, count_ok, ' Images checked for existence.' if check_files_exist else ''))
    print('Total valid items explored:', len(meta_db))
    return meta_db


def generator(meta_db, batch_size):
    """Returns data for keras.fit in form of a list of (X_Train, y_train)"""
    num_samples = len(meta_db)
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_metas = meta_db[offset:offset+batch_size]
            images = []
            angles = []
            for batch_meta in batch_metas:
                image = cv2.imread(batch_meta[0])
                images.append(image)
                angles.append(batch_meta[1])
                # image = image[:,-1,:]  # flip  # TODO
                # images.append(image)
                # angles.append(batch_meta[1])

            X_train = np.array(images)
            y_train = np.array(angles)
            X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
            yield (X_train, y_train)


if __name__ == '__main__':
    meta_db = get_all_meta(1, 5, check_files_exist=True, verbose=False)
