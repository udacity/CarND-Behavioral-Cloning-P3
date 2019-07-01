import numpy
import cv2


RAW_SHAPE = (160, 320, 3)

def preprocess(image):
    assert(image.shape == RAW_SHAPE)

    image = crop(image)
    image = resize(image)
    image = standardize(image)

    return numpy.array(image, dtype='float32')

def crop(image):
    return image[50:-30,:,:]

def resize(image):
    __resize = lambda x: int(x * 0.5)
    shape = image.shape
    resized_wh = (__resize(shape[1]), __resize(shape[0]))
    return cv2.resize(image, resized_wh)

def standardize(image):
    return (image / 255.0 - 0.5) * 2.0

INPUT_SHAPE = preprocess(numpy.empty(RAW_SHAPE)).shape

