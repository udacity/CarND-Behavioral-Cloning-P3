import csv
import cv2
import numpy as np
from PIL import Image

lines = []
with open('data3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data3/IMG/' + filename
    image = cv2.imread(current_path)
#    image = Image.open(current_path)
#    image = image.resize((32, 32), Image.ANTIALIAS)
#    image = np.asarray(image)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

    #augmented data
    images.append(cv2.flip(image, 1))
    measurements.append(measurement*-1)

x_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Conv2D, MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(filters=6, kernel_size=(5, 5),strides=(1,1), padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2), padding='valid'))
model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1,1), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Flatten())
model.add(Dense(units=120,activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')

