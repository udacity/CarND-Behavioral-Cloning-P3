import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
import sys

if __name__ == '__main__':

    lines = []
    with open('simulator_data/simple_3_driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = 'simulator_data/simple_3_IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)

        measurement = float(line[3])
        measurements.append(measurement)

    X_train = np.array(images)
    y_train = np.array(measurements)

    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, nb_epoch=40, validation_split=0.2, shuffle=True)

    model.save('model2.h5')

    sys.exit(0)