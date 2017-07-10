import csv
import os
import cv2
import random
import numpy as np
import h5py
from steering_neural_network import SteeringNeuralNetwork

DATA_DIR = "data"
DATA_H5_PATHS = {
    "TRAIN_PATH": os.path.join(DATA_DIR, "train.h5"),
    "VALID_PATH": os.path.join(DATA_DIR, "valid.h5"),
    "TEST_PATH": os.path.join(DATA_DIR, "test.h5")
}


def create_data_maps(file_path, randomize=True):

    lines = []
    with open(file_path) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # skip the first line
        for line in reader:
            lines.append(line)

    data_mappings = []
    for line in lines:
        # Find the path to the data from the csv file
        source_path = line[0]
        tokens = os.path.split(source_path)
        filename = tokens[-1]
        local_path = os.path.join(DATA_DIR, "IMG", filename)

        # Extract the files names and steering angle mapping
        data_mappings.append((local_path, line[3]))

    if randomize:
        random.shuffle(data_mappings)

    return data_mappings


def split_data_set(data_maps, split_ratio):
    assert (round(sum(split_ratio), 5) == 1.0), "Splitting ratio must add up to 1."

    np.random.shuffle(data_maps)
    num_samples = len(data_maps)

    train_split = int(num_samples * split_ratio[0])
    val_split = train_split + int(num_samples * split_ratio[1])

    train_data_maps = data_maps[0:train_split]
    valid_data_maps = data_maps[train_split:val_split]
    test_data_maps = data_maps[val_split:]

    return train_data_maps, valid_data_maps, test_data_maps


def data_map_generator(data_map, chunk_size=10):
    for chunk_id in range(0, len(data_map), chunk_size):
        data_chunk = data_map[chunk_id:(chunk_id + chunk_size)]
        image_paths_chunk, labels_chunk = zip(*data_chunk)
        yield (image_paths_chunk, np.array(labels_chunk, dtype=float))
        if chunk_id + chunk_size > len(data_map):
            raise StopIteration


def load_images(image_paths, resize_scale=None):
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # resize the image
        if isinstance(resize_scale, (float, int)):
            final_size = (int(resize_scale * img.shape[0]), int(resize_scale * img.shape[1]))
            img = cv2.resize(img, final_size)
        images.append(img)

    return np.array(images)


def save_as_h5_data(data_map, save_to_path):

    chunk_size = 100
    gen = data_map_generator(data_map, chunk_size)

    image_paths_chunk, labels_chunk = next(gen)
    images_chunk = np.asarray(load_images(image_paths_chunk, resize_scale=0.25))
    row_count = images_chunk.shape[0]

    with h5py.File(save_to_path, 'w') as f:
        # Initialize a resizable dataset to hold the output
        images_chunk_maxshape = (None,) + images_chunk.shape[1:]
        labels_chunk_maxshape = (None,) + labels_chunk.shape[1:]

        dset_images = f.create_dataset('images', shape=images_chunk.shape, maxshape=images_chunk_maxshape,
                                       chunks=images_chunk.shape, dtype=images_chunk.dtype)

        dset_labels = f.create_dataset('labels', shape=labels_chunk.shape, maxshape=labels_chunk_maxshape,
                                       chunks=labels_chunk.shape, dtype=labels_chunk.dtype)

        dset_images[:] = images_chunk
        dset_labels[:] = labels_chunk

        for image_paths_chunk, labels_chunk in gen:
            images_chunk = np.asarray(load_images(image_paths_chunk, resize_scale=0.25))

            # Resize the dataset to accommodate the next chunk of rows
            dset_images.resize(row_count + images_chunk.shape[0], axis=0)
            dset_labels.resize(row_count + labels_chunk.shape[0], axis=0)

            # Write the next chunk
            dset_images[row_count:] = images_chunk
            dset_labels[row_count:] = labels_chunk

            # Increment the row count
            row_count += labels_chunk.shape[0]


def read_data(split_ratio=(0.7, 0.2, 0.1)):

    data_exists = True
    for path in DATA_H5_PATHS.values():
        if not os.path.isfile(path):
            data_exists = False
            break

    if data_exists:
        print("Training, validation, and test data were already generated.")

    else:
        data_maps = create_data_maps(os.path.join(DATA_DIR, "driving_log.csv"))
        train_data_map, valid_data_map, test_data_map = split_data_set(data_maps, split_ratio)

        save_as_h5_data(train_data_map, DATA_H5_PATHS["TRAIN_PATH"])
        save_as_h5_data(valid_data_map, DATA_H5_PATHS["VALID_PATH"])
        save_as_h5_data(test_data_map, DATA_H5_PATHS["TEST_PATH"])
        print("Training, validation, and test data are now generated.")

    training_data = h5py.File(DATA_H5_PATHS["TRAIN_PATH"], 'r')
    validation_data = h5py.File(DATA_H5_PATHS["VALID_PATH"], 'r')
    test_data = h5py.File(DATA_H5_PATHS["TEST_PATH"], 'r')

    return training_data, validation_data, test_data

# Load the Data
train_data, valid_data, test_data = read_data()
train_images = train_data["images"]
train_labels = train_data["labels"]
valid_images = valid_data["images"]
valid_labels = valid_data["labels"]
test_images = test_data["images"]
test_labels = test_data["labels"]


# Build the neural network
image_shape = train_images[0].shape
output_shape = 1

steering_network = SteeringNeuralNetwork(image_shape, output_shape)

steering_network.model.compile(optimizer="adam", loss="mse")
steering_network.model.fit(x=train_images, y=train_labels,
                           validation_data=(valid_images, valid_labels),
                           batch_size=64, epochs=500, shuffle="batch")

steering_network.model.save("steering_model.h5")
