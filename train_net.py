import os
#import matplotlib.pyplot as plt
import pandas as pd

from src.utils.general_utils import rebalanced_set, continuous_to_bins, generate_data_from, create_train_val_from

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Lambda

csv_file_name='driving_log.csv'
data_path = 'data'
model_path = 'logs'
model_name = 'LeNet_DropOut.h5'
input_img_shape = [160, 320, 3]
batch_size = 128
n_bins = 7
val_portion = 0.1

desriptor = pd.read_csv(os.path.join(data_path, csv_file_name))

# Utility functions to:
# create paths to images
create_paths_to_images = lambda x: [os.path.join(data_path, v) for v in x]
# split the dataset into training and validation data

#n, bins, patches = plt.hist(desriptor.steering, bins=n_bins, color='grey')

#plt.xlabel('Classes')
#plt.ylabel('Samples')
#plt.show()

Y_train_binned = continuous_to_bins(desriptor.steering, n_bins=n_bins)
binned_indices = rebalanced_set(Y_train_binned)

train_indices, val_indices = create_train_val_from(binned_indices, portion_of_val_set=val_portion)
print("Training set size: {}, Validation set size: {}".format(len(train_indices), len(val_indices)))


model = Sequential()
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=input_img_shape))
model.add(Convolution2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(16, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(32, (3, 3), activation="relu"))
model.add(Convolution2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.25))
model.add(Dense(84))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.summary() #prints a summary representation of your model.
model_config = model.get_config()
#model = Sequential.from_config(model_config)

paths_to_images = create_paths_to_images(desriptor.center)


model.fit_generator(generate_data_from(train_indices, paths_to_images, desriptor.steering, batch_size),
                    samples_per_epoch=len(train_indices)//batch_size,
                    nb_epoch=4,
                    validation_data=generate_data_from(train_indices, paths_to_images, desriptor.steering, batch_size),
                    nb_val_samples=len(val_indices)//batch_size)



model.save(os.path.join(model_path, model_name))



