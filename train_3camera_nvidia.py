import csv
import cv2
import numpy as np
from PIL import Image
import os

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import sklearn
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


batch_size = 32
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)


#images = []
#measurements = []
#for line in lines:
##    source_path = line[0]
##    filename = source_path.split('/')[-1]
##    current_path = 'data/' + filename
##    image = cv2.imread(current_path)
#    path = 'data/'
#    img_center = images.append(np.asarray(Image.open(path+line[0])))
#    img_left = images.append(np.asarray(Image.open(path+line[1].replace(' ',''))))
#    img_right = images.append(np.asarray(Image.open(path+line[2].replace(' ','')))) 
##    image = Image.open(current_path)
##    image = image.resize((32, 32), Image.ANTIALIAS)
##    image = np.asarray(image)
##    images.append(image)
#    correction = 0.1
#    steering_center = float(line[3])
#    steering_left = steering_center + correction
#    steering_right = steering_center - correction
#    measurements.append(steering_center)
#    measurements.append(steering_left)
#    measurements.append(steering_right)
#    
#    #augmented data
##    images.append(cv2.flip(image_center, 1))
##    measurements.append(steering_center*-1)
#
#x_train = np.array(images)
#y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Conv2D, MaxPooling2D
from keras.layers import Cropping2D
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

from matplotlib.pyplot import savefig

model = Sequential()
model.add(Cropping2D(cropping=((50,20),(0,0)),input_shape = (160,320,3)))
#50 rows pixels from the top of the image
#20 rows pixels from the bottom of the image
#0 columns of pixels from the left of the image
#0 columns of pixels from the right of the image
model.add(Lambda(lambda x: x/255.0 - 0.5 ))

model.add(Conv2D(filters=24, kernel_size=(5, 5),strides=(2,2), padding='valid',activation='relu'))

#model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2), padding='valid'))
model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2,2), padding='valid', activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2,2), padding='valid', activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))

model.add(Flatten())
model.add(Dense(units=100,activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

#model.save('model.h5')

history_object = model.fit_generator(train_generator, steps_per_epoch = len(train_samples)/batch_size ,
    validation_data = validation_generator, validation_steps = len(validation_samples)/batch_size,
    nb_epoch=5, verbose=1)

print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

plt.savefig('loss.jpg')

model.save('model.h5')
