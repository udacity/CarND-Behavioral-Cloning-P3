from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, Cropping2D, Lambda, Activation, Dropout
from keras.initializers import glorot_normal

def nvidia():
  model = Sequential()
  model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
  model.add(Cropping2D(cropping=((70,25), (0,0))))
  
  model.add(Conv2D(
    24, 5, strides=1, padding='valid', 
    kernel_initializer=glorot_normal(seed=1), bias_initializer='zeros'
  ))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Conv2D(
    36, 5, strides=1, padding='valid',
    kernel_initializer=glorot_normal(seed=1), bias_initializer='zeros'
  ))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Conv2D(
    48, 5, strides=1, padding='valid',
    kernel_initializer=glorot_normal(seed=1), bias_initializer='zeros'
  ))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Conv2D(
    64, 3, padding='valid',
    kernel_initializer=glorot_normal(seed=1), bias_initializer='zeros'
  ))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Conv2D(
    64, 3, padding='valid',
    kernel_initializer=glorot_normal(seed=1), bias_initializer='zeros'
  ))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Flatten())

  model.add(Dense(100, kernel_initializer=glorot_normal(seed=1), bias_initializer='zeros'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Dense(50, kernel_initializer=glorot_normal(seed=1), bias_initializer='zeros'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Dense(10, kernel_initializer=glorot_normal(seed=1), bias_initializer='zeros'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Dense(1, kernel_initializer=glorot_normal(seed=1), bias_initializer='zeros'))
  
  return model