import numpy as np
import cv2, os
from sklearn.utils import shuffle

create_paths_to_images = lambda x, data_path: np.array([os.path.join(data_path, v) for v in x])

read_rgb_img = lambda path: cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def ensure_valid_values(paths, measure, dtype=np.float64):
    paths = np.array(paths)
    measure = np.array(measure)
    new_paths, new_measure = [],[]
    for p, m in zip(paths, measure):
        if os.path.exists(p) and (type(m)==dtype):
            new_paths.append(p), new_measure.append(m)
        else:
            print("Incorrect path:", p, m, type(m).__name__)
    new_paths = np.array(new_paths)
    new_measure = np.array(new_measure)
    assert len(new_paths)>0, "Provided incorrect paths. No paths or measure will be generated."

    return new_paths, new_measure

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



def generate_data_with_augmentation_from(paths,
                                         measurements,
                                         batch_sz=32,
                                         random_flip=True):

    epoch_sz = len(measurements)

    horizontal_flip = lambda x, m: (cv2.flip(x, 0), -m)
    no_flip = lambda x, m: (x, m)

    def augmentation(p,m):

        im, m = read_rgb_img(p), m

        if random_flip:
            flip_choice = np.random.choice([horizontal_flip, no_flip])
            im, m = flip_choice(im, m)

        return im, m

    while True:
        paths, measurements = shuffle(paths, measurements)
        if batch_sz is None:
            for p, m in zip(paths, measurements):
                im, m = augmentation(p, m)
                yield (im, m)
        else:
            for start, end in zip(range(0, epoch_sz, batch_sz),
                                  range(batch_sz, epoch_sz + 1, batch_sz)):
                ims, ms = [],[]
                for p,m in zip(paths[start:end], measurements[start:end]):
                    im, m = augmentation(p,m)
                    ims.append(im)
                    ms.append(m)
                yield (np.array(ims), np.array(ms))





if __name__=='__main__':

    rand_range = np.random.rand(100)-0.5
    rand_range[10:20]=0.0
    bins = continuous_to_bins(rand_range)

    indices = rebalanced_set(bins)

    for r,b in zip(rand_range, bins):
        print(r,b)
