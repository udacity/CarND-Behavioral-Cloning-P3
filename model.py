import numpy as np
import keras
import tensorflow as tf
import csv
import cv2
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(csvfile)
    for line in reader:
        lines.append(line)
        
train_samples, validation_samples = train_test_split(lines, test_size=0.2, random_state=42)

def generator(samples, batch_size = 32, correction = 0.2):
    num_samples = len(samples)
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                #Center image
                name = batch_sample[0].split('/')[-1]
                current_path = 'data/IMG/' + name
                center_image = cv2.imread(current_path)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_image = cv2.resize(center_image, (160, 80))
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                #Left image
                name = batch_sample[1].split('/')[-1]
                current_path = 'data/IMG/' + name
                left_image = cv2.imread(current_path)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_image = cv2.resize(left_image, (160, 80))
                left_angle = center_angle + correction
                images.append(left_image)
                angles.append(left_angle)
                
                #Right image
                name = batch_sample[2].split('/')[-1]
                current_path = 'data/IMG/' + name
                right_image = cv2.imread(current_path)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                right_image = cv2.resize(right_image, (160, 80))
                right_angle = center_angle - correction
                images.append(right_image)
                angles.append(right_angle)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32, correction=0.2)
validation_generator = generator(validation_samples, batch_size=32, correction=0.2)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Reshape, Cropping2D, Conv2D

model = Sequential()
model.add(Cropping2D(cropping=((25, 12), (0, 0)), input_shape=(80, 160, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), 
                 padding='same', activation='relu'))
model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2),
                 padding='same', activation='relu'))
model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2),
                 padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1),
                 padding='valid', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1),
                 padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples),
                    epochs=1, verbose=1)

model.save('model.h5')

