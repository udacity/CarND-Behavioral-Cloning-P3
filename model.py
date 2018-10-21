import csv
import cv2
import numpy as np
from pathlib import Path
from sklearn import utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def get_model():
    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), padding='valid', activation='relu'))
    model.add(Convolution2D(64, (3, 3), padding='valid', activation='relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for pos in range(3):
                    file_name = Path(batch_sample[pos]).name
                    image_path = Path('./data/IMG').joinpath(file_name)
                    images.append(cv2.imread(image_path.as_posix()))

                center_angle = float(batch_sample[6])
                correction = 0.2
                angles.extend([
                    center_angle,
                    center_angle + correction,
                    center_angle - correction
                ])

            for i in range(len(images)):
                images.append(cv2.flip(images[i], 1))
                angles.append(-angles[i])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield utils.shuffle(X_train, y_train)

samples = []
with Path('./data/driving_log.csv').open() as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, valid_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
valid_generator = generator(valid_samples, batch_size=32)

model = get_model()
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=len(train_samples),
                    validation_data=valid_generator,
                    validation_steps=len(valid_samples), epochs=2)

model.save('model.h5')
