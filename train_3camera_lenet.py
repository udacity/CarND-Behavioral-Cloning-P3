import csv
import cv2
import numpy as np
from PIL import Image

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
#    source_path = line[0]
#    filename = source_path.split('/')[-1]
#    current_path = 'data/' + filename
#    image = cv2.imread(current_path)
    path = 'data/'
    img_center = images.append(np.asarray(Image.open(path+line[0])))
    img_left = images.append(np.asarray(Image.open(path+line[1].replace(' ',''))))
    img_right = images.append(np.asarray(Image.open(path+line[2].replace(' ','')))) 
#    image = Image.open(current_path)
#    image = image.resize((32, 32), Image.ANTIALIAS)
#    image = np.asarray(image)
#    images.append(image)
    correction = 0.2
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)
    
    #augmented data
#    images.append(cv2.flip(image_center, 1))
#    measurements.append(steering_center*-1)

x_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Conv2D, MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20),(0,0)),input_shape = (160,320,3)))
#50 rows pixels from the top of the image
#20 rows pixels from the bottom of the image
#0 columns of pixels from the left of the image
#0 columns of pixels from the right of the image
model.add(Lambda(lambda x: x/255.0 - 0.5 ))
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

