import numpy as np
import cv2
from sklearn.utils import shuffle


def continuous_to_bins(vector, n_bins=9):
    vector = np.array(vector)
    shape = vector.shape
    vector = np.reshape(vector, [-1])
    range = vector.max()-vector.min()
    step = range/float(n_bins-1)
    binned = np.round((vector - vector.min())/step, 0)
    binned = np.reshape(binned, shape)
    return binned.astype(int)


def rebalanced_set(class_lables):
    """
    
    :param class_lables: 
    :return: index of rebalanced dataset
    """

    class_lables = np.array(class_lables)

    train_class_indexes = [[] for _ in range(class_lables.max() + 1)]
    n_samples = np.zeros(class_lables.max() + 1)
    for i, l in enumerate(class_lables):
        train_class_indexes[l].append(i)
        n_samples[l] += 1

    for i, l in enumerate(train_class_indexes):
        size = len(l)
        for _ in range(int(n_samples.max()) - size):
            train_class_indexes[i].append(l[np.random.randint(size)])

    train_class_indexes = np.reshape(np.array(train_class_indexes, dtype=np.int), [-1])
    train_class_indexes = shuffle(train_class_indexes)

    return train_class_indexes



def generate_data_from(rebalanced_index, paths_to_images, measurements, batch_sz=None):
    
    epoch_sz = len(rebalanced_index)
    paths_to_images = np.array(paths_to_images)
    measurements = np.array(measurements)

    read_img = lambda x: cv2.imread(paths_to_images[x])

    while True:
        rebalanced_index = shuffle(rebalanced_index)
        if batch_sz is None:
            for i in rebalanced_index:
                yield (np.array(read_img(i)), measurements[i])
        else:
            for start, end in zip(range(0, epoch_sz, batch_sz),
                                  range(batch_sz, epoch_sz+1, batch_sz)):
                yield (np.array([read_img(i) for i in rebalanced_index[start:end]]),
                       measurements[rebalanced_index[start:end]])

def create_train_val_from(dataset, portion_of_val_set=0.2):
    split = int(len(dataset)*portion_of_val_set)
    dataset = shuffle(dataset)

    return dataset[split:], dataset[:split]





if __name__=='__main__':

    rand_range = np.random.rand(100)-0.5
    rand_range[10:20]=0.0
    bins = continuous_to_bins(rand_range)

    indices = rebalanced_set(bins)

    for r,b in zip(rand_range, bins):
        print(r,b)
