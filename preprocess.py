import numpy
import cv2


RAW_SHAPE = (160, 320, 3)

def preprocess(image):
    assert(image.shape == RAW_SHAPE)

    # Crop
    image = image[50:-30,:,:]

    # Resize
    resize = lambda x: int(x * 0.5)
    shape = image.shape
    resized_wh = (resize(shape[1]), resize(shape[0]))
    image = cv2.resize(image, resized_wh)

    # Standardize
    image = (image / 255.0 - 0.5) * 2.0

    return numpy.array(image, dtype='float32')

INPUT_SHAPE = preprocess(numpy.empty(RAW_SHAPE)).shape

