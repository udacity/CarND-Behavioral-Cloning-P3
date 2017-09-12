import csv
import os
import cv2
from skimage.util import random_noise
import random
import numpy as np
import h5py


def read_data_set(data_dir, csv_file_name):
    data_set = []

    lines = []
    file_path = os.path.join(data_dir, csv_file_name)
    with open(file_path) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # skip the first line
        for line in reader:
            lines.append(line)

    for line in lines:

        # don't add if speed is too low
        speed = float(line[6])
        if speed < 0.1:
            continue

        angle = float(line[3])

        # Find the path to the data from the csv file
        # center, left, and right image path
        for i in range (0, 3):
            source_path = line[i]
            tokens = os.path.split(source_path)
            filename = tokens[-1]
            local_path = os.path.join(data_dir, "IMG", filename)
            if i == 1: # Left
                modified_angle = angle + 0.25
            elif i == 2:  # Right
                modified_angle = angle - 0.25
            else:  # Center
                modified_angle = angle

            # don't add if angle is straight "remove_percent" of times
            remove_percent = 0.70
            if -0.25 < modified_angle < 0.25 and random.random() < remove_percent:
                pass
            else:
                data_set.append((local_path, modified_angle))

        random.shuffle(data_set)
    return data_set


def split_data_set(data_set, split_ratio):
    assert (round(sum(split_ratio), 5) == 1.0), "Splitting ratio must add up to 1."

    np.random.shuffle(data_set)
    num_samples = len(data_set)

    train_split = int(num_samples * split_ratio[0])
    val_split = train_split + int(num_samples * split_ratio[1])

    train_data_maps = data_set[0:train_split]
    valid_data_maps = data_set[train_split:val_split]
    test_data_maps = data_set[val_split:]

    return train_data_maps, valid_data_maps, test_data_maps


def load_images(image_paths, resize_scale=None):
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        img = random_noise(img, mode="s&p", amount=0.05)
        #cv2.imshow("img", img)
        #cv2.waitKey(0)
        images.append(img)

    return np.array(images)


def create_data_sets(data_dirs, split_ratio=(0.7, 0.2, 0.1)):

    data_sets = []
    for data_dir in data_dirs:
        data_sets += read_data_set(data_dir, "driving_log.csv")
    random.shuffle(data_sets)

    train_data_set, valid_data_set, test_data_set = split_data_set(data_sets, split_ratio)
    return train_data_set, valid_data_set, test_data_set


def shuffle_data(x, y):
    zipped_data = list(zip(x, y))
    random.shuffle(zipped_data)
    return list(zip(*zipped_data))


def preprocess_image(img):
    '''
    Method for preprocessing images: this method is the same used in drive.py, except this version uses
    BGR to YUV and drive.py uses RGB to YUV (due to using cv2 to read the image here, where drive.py images are
    received in RGB)
    '''
    # Crop the image to focus on the road
    new_img = img[55:135, :, :]
    # apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3, 3), 0)
    # scale to 64x64x3
    new_img = cv2.resize(new_img,(64, 64), interpolation=cv2.INTER_AREA)

    return new_img


def data_generator(data_set, batch_size=64, add_noise=False):
    random.shuffle(data_set)
    image_paths, angles = list(zip(*data_set))
    X = []
    y = []
    while True:
        for i in range(len(angles)):
            img = cv2.imread(image_paths[i])
            angle = angles[i]

            img = preprocess_image(img)
            if add_noise and random.random() < 0.5:
                img = random_noise(img, mode="s&p", amount=0.1)
            #cv2.imshow("img", img)
            #cv2.waitKey(0)

            X.append(img)
            y.append(angle)

            if len(X) == batch_size:
                # yield when batch size has been reached
                yield (np.array(X), np.array(y))

                # Remove the batch from the buffered array
                X = []
                y = []

                # Reshuffle the data set
                random.shuffle(data_set)
                image_paths, angles = list(zip(*data_set))

            # flip horizontally and invert steer angle, if magnitude is > 0.33
            if abs(angle) > 0.33:
                img = cv2.flip(img, 1)
                angle *= -1
                X.append(img)
                y.append(angle)

                if len(X) == batch_size:
                    # yield when batch size has been reached
                    yield (np.array(X), np.array(y))

                    # Remove the batch from the buffered array
                    X = []
                    y = []

                    # Reshuffle the data set
                    random.shuffle(data_set)
                    image_paths, angles = list(zip(*data_set))
                    

def train_model(data_dir_paths, model_name="steering_model.h5"):

    new_model = False

    # Load the Data
    BATCH_SIZE = 64
    train_data_sets, valid_data_sets, test_data_sets = create_data_sets(data_dir_paths, split_ratio=(0.8, 0.2, 0.0))
    train_generator = data_generator(train_data_sets, batch_size=BATCH_SIZE, add_noise=True)
    valid_generator = data_generator(valid_data_sets, batch_size=BATCH_SIZE)
    test_generator = data_generator(test_data_sets, batch_size=BATCH_SIZE)
    print("Number of training sample = %d" % len(train_data_sets))

    # Build the neural network
    image_shape = (64, 64, 3)  # (160, 320, 3)
    output_shape = 1

    from keras.models import load_model
    from keras.callbacks import ModelCheckpoint
    from steering_neural_network import SteeringNeuralNetwork

    model = None
    if not new_model:
        # Load partly trained model
        model = load_model(model_name)

    steering_network = SteeringNeuralNetwork(image_shape, output_shape, curr_model=model)
    steering_network.model.compile(optimizer="adam", loss="mse")
    checkpoint = ModelCheckpoint('steering_model{epoch:02d}.h5')
    steering_network.model.fit_generator(train_generator, steps_per_epoch=len(train_data_sets)/BATCH_SIZE,
                                         validation_data=valid_generator, validation_steps=len(valid_data_sets)/BATCH_SIZE,
                                         epochs=10, verbose=1, callbacks=[checkpoint])

    steering_network.model.save("steering_model.h5")

    # Test the model
    loss = steering_network.model.evaluate_generator(test_generator, steps=len(test_data_sets)/BATCH_SIZE)
    print("\nTest Loss: %4f" % loss)


# List out all the folders with data
udacity_data_dir = "data/udacity_data"
new_data_dirs = [
    #"data/170609_data/JungleTrack",
    "data/170609_data/LakeTrack",
    #"data/170727_data/JungleTrack",
    "data/170727_data/LakeTrack"
]

train_data_dir = [udacity_data_dir]

for new_data_dir in new_data_dirs:
    for dated_data_dir in os.listdir(new_data_dir):
        train_data_dir.append(os.path.join(new_data_dir, dated_data_dir))


train_model(data_dir_paths=train_data_dir)

#load_images(["C:\\Projects\\Udacity\\Term1\\Behaviour Cloning\\CarND-Behavioral-Cloning-P3\\data\\170609_data\\LakeTrack\\2017_06_09_10_34_53_854\\IMG\\center_2017_06_09_10_33_52_319.jpg"])