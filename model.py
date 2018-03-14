from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, Cropping2D, Lambda, Activation, Dropout

def LeNet(input_shape, mu, sigma, dropout=1.0):
  model = Sequential()
  model.add(Lambda(lambda x: (x - mu)/sigma, input_shape=input_shape)) # preprocess, normalization
  model.add(Cropping2D(cropping=((75,25), (0,0))))
  
  # layer 1
  model.add(Conv2D(6, 5, strides=1, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Dropout(dropout, seed=1))
  
  # layer 2
  model.add(Conv2D(16, 5, strides=1, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Dropout(dropout, seed=1))
  
  # layer 3
  model.add(Conv2D(32, 5, strides=1, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Dropout(dropout, seed=1))
  
  model.add(Flatten())

  model.add(Dense(200, activation='relu'))
  model.add(Dropout(dropout, seed=1))
  
  model.add(Dense(120, activation='relu'))
  model.add(Dropout(dropout, seed=1))
  
  model.add(Dense(84, activation='relu'))
  model.add(Dropout(dropout, seed=1))
  
  model.add(Dense(1))
  
  return model

def nvidia(input_shape, mu, sigma, dropout=1.0):
  model = Sequential()
  model.add(Lambda(lambda x: (x - mu)/sigma, input_shape=input_shape)) # preprocess, normalization
  model.add(Cropping2D(cropping=((35,12), (0,0))))
  
  model.add(Conv2D(24, 5, strides=1, padding='valid'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Conv2D(36, 5, strides=1, padding='valid'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Conv2D(48, 5, strides=1, padding='valid'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Conv2D(64, 3, padding='valid'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Conv2D(64, 3, padding='valid'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Flatten())

  model.add(Dense(100, activation='relu'))
  model.add(Dropout(dropout))
  model.add(Dense(50, activation='relu'))
  model.add(Dropout(dropout))
  model.add(Dense(10, activation='relu'))  
  model.add(Dropout(dropout))
  model.add(Dense(1))
  
  return model