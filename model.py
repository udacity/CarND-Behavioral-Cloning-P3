import os
import csv
import cv2
from scipy import ndimage
import numpy as np


def read_driving_data_csv( base_dir_names ):
    lines = []
    for base_dir in base_dir_names:
        with open(base_dir + 'driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                if ( line[0] != 'center'):
                    lines.append(line)
    return lines

def load_data_from_lines( lines, base_dirs ):
    images = []
    measurements = []
    for line in lines:
        measurement_correction_factor = [0., 0.2, -0.2 ]
        for i in range(3):
            source_path = line[i]
            filename = os.path.basename(source_path)
            for base_dir in base_dirs:
                current_path = base_dir + 'IMG/' + filename
                if os.path.exists( current_path):
                    image = ndimage.imread(current_path)
                    images.append(image)
                    images.append(cv2.flip(image,1))
                    measurement = float(line[3])
                    measurements.append((measurement + measurement_correction_factor[i]))
                    measurements.append((measurement + measurement_correction_factor[i])*-1.0)
                    break
    return images, measurements


base_dirs = ['sample_data/data/', 'mydata/' ]

lines = read_driving_data_csv( base_dirs )
print("=== read line : {:}".format(len(lines)))

images, measurements = load_data_from_lines( lines, base_dirs )
print("=== images : {:}".format(len(images)))
    


X_train = np.array(images)
y_train = np.array(measurements)
############################################################
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda( lambda x: x / 255.0 -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')

          
