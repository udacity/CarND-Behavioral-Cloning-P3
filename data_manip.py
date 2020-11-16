import cfg
import cv2
import numpy as np
import matplotlib.pyplot as plt
from reader import read_sim_data
import random


def flip(images, measurements):
    #fig, (ax1, ax2) = plt.subplots(1,2)
    #ax1.imshow(images[0])
    images = np.flip(images, axis=-2)
    #ax2.imshow(images[0])
    #plt.show()
    measurements = -1.0 * measurements
    return images, measurements


def show_example(images, measurements, start_index, columns=3, second_row_offset=3):
    _, axes = plt.subplots(2, columns, figsize=(12, 9))
    for x in range(columns):
        idx = start_index + x
        axes[0][x].title.set_text(measurements[idx])
        axes[0][x].imshow(images[idx])
        idx += second_row_offset
        axes[1][x].title.set_text(measurements[idx])
        axes[1][x].imshow(images[idx])
    plt.show()


def preprocess(X, y, verbose=False):
    print('Shapes before preprocess: X: {}, y: {}'.format(X.shape, y.shape))

    # before
    sample_idx = random.randrange(len(X))
    sample_img_before = np.array(X[sample_idx])
    sample_measure_before = y[sample_idx]

    # preprocess
    X_flipped, y_flipped = flip(X, y)
    X_new = np.vstack((X, X_flipped))
    y_new = np.hstack((y, y_flipped))
    print('Shapes after preprocess: X: {}, y: {}'.format(X_new.shape, y_new.shape))

    # show before vs. after
    if verbose:
        sample_img_after = np.array(X_new[sample_idx + len(X)])
        sample_measure_after = y_new[sample_idx + len(y)]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        ax1.imshow(sample_img_before)
        ax1.title.set_text(str(sample_measure_before))
        ax2.imshow(sample_img_after)
        ax2.title.set_text(str(sample_measure_after))
        plt.show()

    return X_new, y_new


def test():
    X, y = read_sim_data(1)
    X, y = preprocess(X, y, True)


if __name__ == '__main__':
    test()
