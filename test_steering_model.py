from keras.models import load_model
import os
import h5py
import math

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
    steering_angle = float(model.predict(test_images[i][None, :, :, :], batch_size=1))
    error = test_labels[i] - steering_angle
    errors.append(error)
    if i % 10 == 0:
        print("Steering angle: %.4f    Error: %.5f" % (steering_angle, error))


mse = math.sqrt(sum([error**2 for error in errors]))
print("MSE: %.5f" % mse)