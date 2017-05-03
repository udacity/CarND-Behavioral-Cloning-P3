from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D
from keras.layers.normalization import BatchNormalization


def SimplifiedModelExtraDropout(FLAGS, input_img_shape):

    assert FLAGS.top_crop is not None, "FLAGS.top_crop wasn't provided"
    assert FLAGS.bottom_crop is not None, "FLAGS.bottom_crop wasn't provided"
    assert FLAGS.width is not None, "FLAGS.width wasn't provided"


    model = Sequential()
    model.add(Cropping2D(cropping=((FLAGS.top_crop,FLAGS.bottom_crop),(0,0)), input_shape=input_img_shape))
    model.add(Lambda(lambda x: (x/255.0)-0.5))
    model.add(Convolution2D(int(16 * FLAGS.width), 3, 3, border_mode='same', activation="elu"))
    model.add(Convolution2D(int(16 * FLAGS.width), 3, 3, subsample=(2,2), activation="elu"))
    model.add(BatchNormalization())
    model.add(Convolution2D(int(32 * FLAGS.width), 3, 3, border_mode='same', activation="elu"))
    model.add(Convolution2D(int(32 * FLAGS.width), 3, 3, subsample=(2,2), activation="elu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Convolution2D(int(64 * FLAGS.width), 3, 3, border_mode='same', activation="elu"))
    model.add(Convolution2D(int(64 * FLAGS.width), 3, 3, subsample=(2,2), activation="elu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Convolution2D(int(64 * FLAGS.width), 3, 3, border_mode='same', activation="elu"))
    model.add(Convolution2D(int(64 * FLAGS.width), 3, 3, subsample=(2,2), activation="elu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Convolution2D(int(128 * FLAGS.width), 3, 3, border_mode='same', activation="elu"))
    model.add(Convolution2D(int(128 * FLAGS.width), 3, 3, subsample=(2,2), activation="elu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(16, activation='elu'))
    model.add(Dense(1, activation='linear'))


    return model
