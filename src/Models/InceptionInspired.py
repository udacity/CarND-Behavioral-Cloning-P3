from keras.models import Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image

def InceptionInspired(FLAGS, input_img_shape):



    assert FLAGS.top_crop is not None, "FLAGS.top_crop wasn't provided"
    assert FLAGS.bottom_crop is not None, "FLAGS.bottom_crop wasn't provided"
    assert FLAGS.width is not None, "FLAGS.width wasn't provided"

    X = Input(shape=input_img_shape)
    X = Lambda(lambda x: (x / 255.0) - 0.5)(X)
    axis = bn_axis, scale = False
    BatchNormalization(axis=-1, scale=False)




