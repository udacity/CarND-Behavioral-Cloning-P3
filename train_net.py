import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.utils.general_utils import rebalanced_set, continuous_to_bins, generate_driving_data_from
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Lambda

tf.app.flags.DEFINE_string('data_location',
                           'data',
                           'Define the location of the data folder containing csv descriptor and IMG folder - Default: data')
tf.app.flags.DEFINE_string('logs_location',
                           'logs',
                           'Define the location of the logs folder. It will be used for storing models - Default: logs')
tf.app.flags.DEFINE_string('descriptor_name',
                           'driving_log.csv',
                           'Provide the name of the data descriptor - Default: driving_log.csv')
tf.app.flags.DEFINE_string('model_name',
                           'LeNet_DropOut.h5',
                           'Provide the name of the data descriptor - Default: LeNet_DropOut.h5')
tf.app.flags.DEFINE_integer('batch_size',512,'Provide the batch size - Default: 512')
tf.app.flags.DEFINE_float('val_portion', 0.15, 'Define the portion of the dataset used for validation')
FLAGS = tf.app.flags.FLAGS
csv_file_name=FLAGS.descriptor_name
data_path = FLAGS.data_location
model_path = FLAGS.logs_location
model_name = FLAGS.model_name
batch_size = FLAGS.batch_size
val_portion = FLAGS.val_portion

input_img_shape = [160, 320, 3]
n_bins = 7

descriptor = pd.read_csv(os.path.join(data_path, csv_file_name))

"""
import matplotlib.pyplot as plt
n, bins, patches = plt.hist(descriptor.steering, bins=n_bins, color='grey')
plt.xlabel('Classes')
plt.ylabel('Samples')
plt.show()
"""

Y_train_binned = continuous_to_bins(descriptor.steering, n_bins=n_bins)
binned_indices = rebalanced_set(Y_train_binned)

train_indices, val_indices = train_test_split(binned_indices, test_size=val_portion)
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

model.fit_generator(generate_driving_data_from(train_indices, descriptor, batch_size, data_path),
                    samples_per_epoch=len(train_indices)//batch_size,
                    nb_epoch=5,
                    validation_data=generate_driving_data_from(val_indices, descriptor, batch_size, data_path),
                    nb_val_samples=len(val_indices)//batch_size)



model.save(os.path.join(model_path, model_name))



