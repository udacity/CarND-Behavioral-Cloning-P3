from keras.models import load_model
import os
import h5py
import math
import cv2


def preprocess_image(img):
    '''
    Method for preprocessing images: this method is the same used in drive.py, except this version uses
    BGR to YUV and drive.py uses RGB to YUV (due to using cv2 to read the image here, where drive.py images are
    received in RGB)
    '''
    # original shape: 160x320x3, input shape for neural net: 66x200x3
    new_img = img[50:140,:,:]
    # apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3,3), 0)
    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    return new_img


def read_test_data(data_dir):
    test_path =  os.path.join(data_dir, "test.h5")
    test_data = h5py.File(test_path, 'r')

    return test_data

test_data = read_test_data("data/udacity_data")
test_images = test_data["images"]
test_labels = test_data["labels"]

model = load_model("steering_model.h5")
errors = []
for i in range(len(test_labels)):
    steering_angle = float(model.predict(preprocess_image(test_images[i])[None, :, :, :], batch_size=1))
    error = test_labels[i] - steering_angle
    errors.append(error)
    if i % 10 == 0:
        print("Steering angle: %.4f    Error: %.5f" % (steering_angle, error))


mse = math.sqrt(sum([error**2 for error in errors]))
print("MSE: %.5f" % mse)