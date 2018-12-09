import os
import csv
import cv2
from scipy import ndimage
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn


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

#images, measurements = load_data_from_lines( lines, base_dirs )
#print("=== images : {:}".format(len(images)))
    

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
print("train_samples: {:}, validation_samples: {:}".format(len(train_samples),len(validation_samples) ))

def generator(samples, batch_size=32):
    measurement_correction_factor = [0., 0.2, -0.2 ]
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples_shuffled = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples_shuffled[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3): # number 0 = cener , 1 = left, 2 = right ( CVS column number )
                    source_path = batch_sample[i]
                    filename = os.path.basename(source_path)
                    for base_dir in base_dirs:
                        current_path = base_dir + 'IMG/' + filename
                        if os.path.exists( current_path):
                            image = ndimage.imread(current_path)
                            images.append(image)
                            images.append(cv2.flip(image,1))
                            angle = float(batch_sample[3])
                            angles.append((angle + measurement_correction_factor[i]))
                            angles.append((angle + measurement_correction_factor[i])*-1.0)
                            break

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)


#model.compile(loss='mse', optimizer='adam')
#model.fit_generator(train_generator, samples_per_epoch= /
#            len(train_samples), validation_data=validation_generator, /
#            nb_val_samples=len(validation_samples), nb_epoch=3)

"""
If the above code throw exceptions, try 
model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
"""

#X_train = np.array(images)
#y_train = np.array(measurements)
############################################################
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt

keep_prob=0.5

model = Sequential()
model.add(Lambda( lambda x: x / 255.0 -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(keep_prob))
model.add(Dense(50))
model.add(Dropout(keep_prob))
model.add(Dense(10))
model.add(Dropout(keep_prob))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.summary()

#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()         

model.save('model.h5')
