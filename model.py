import csv
import cv2
import numpy as np
from pathlib import Path

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def return_model():
    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
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

lines = []
csv_path = Path('./data/driving_log.csv')
with csv_path.open() as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
angles = []
for line in lines:
    for pos in range(3):
        file_name = Path(line[pos]).name
        image_path = Path('./data/IMG').joinpath(file_name)
        images.append(cv2.imread(image_path.as_posix()))

    steering_center = float(line[3])
    correction = 0.2
    angles.extend([
        steering_center,
        steering_center + correction,
        steering_center - correction
    ])

assert(len(images) == len(angles))

for i in range(len(images)):
    images.append(cv2.flip(images[i], 1))
    angles.append(-angles[i])

X_train = np.array(images)
y_train = np.array(angles)

model = return_model()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')
