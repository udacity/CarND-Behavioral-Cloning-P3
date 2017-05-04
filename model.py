import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.utils.general_utils import rebalanced_set, \
    continuous_to_bins, generate_data_with_augmentation_from,\
    create_paths_to_images, ensure_valid_values
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as K

from src.Models.SimplifiedModel import SimplifiedModel
from src.Models.SimplifiedModel_Extra_Dropout import SimplifiedModelExtraDropout

tf.flags.DEFINE_string('data_location',
                       'data',
                       'Define the location of the data folder containing csv descriptor and IMG folder - Default: data')
tf.flags.DEFINE_string('logs_location',
                       'logs',
                       'Define the location of the logs folder. It will be used for storing models - Default: logs')
tf.flags.DEFINE_string('descriptor_name',
                       'driving_log.csv',
                       'Provide the name of the data descriptor - Default: driving_log.csv')
tf.flags.DEFINE_string('model_type',
                       'SimplifiedModelExtraDropout',
                       'Provide the name of the net architecture to be used for training [SimplifiedModel,SimplifiedModelExtraDropout] - Default: SimplifiedModel')
tf.flags.DEFINE_string('model_name',
                       'LeNet_DropOut.h5',
                       'Provide the name of the data descriptor - Default: LeNet_DropOut.h5')
tf.flags.DEFINE_integer('batch_size',256,
                        'Provide the batch size - Default: 256')
tf.flags.DEFINE_integer('epochs',5,
                        'Specify the number of epochs for the training - Default: 5')
tf.flags.DEFINE_integer('bins',5,
                        'Specify the number of bins used to rebalance the data - Default: 5')
tf.flags.DEFINE_integer('top_crop',0,
                        'Specify the number pixels to be cropped form the top - Default: 0')
tf.flags.DEFINE_integer('bottom_crop',20,
                        'Specify the number pixels to be cropped on the bottom - Default: 20')
tf.flags.DEFINE_float('val_portion', 0.15,
                      'Define the portion of the dataset used for validation')
tf.flags.DEFINE_float('shift_value', 0.20,
                      'Define the shift value for cameras - Default: 0.20')
tf.flags.DEFINE_float('width', 1.0,
                      'Define the width scaller for the net. Default: 1.0 (float)')
tf.flags.DEFINE_bool('shift', True, "Camera shift augmentation is set for True. Set for False to turn off.")
tf.flags.DEFINE_bool('flip', True, "Camera flip augmentation is set for True. Set for False to turn off.")

FLAGS = tf.flags.FLAGS
csv_file_name=FLAGS.descriptor_name
data_path = FLAGS.data_location
model_path = FLAGS.logs_location
model_name = FLAGS.model_name
batch_size = FLAGS.batch_size

input_img_shape = [160, 320, 3]

descriptor = pd.read_csv(os.path.join(data_path, csv_file_name))

if FLAGS.shift:
    train_steering,val_steering, train_paths_center,val_paths, train_paths_left, _, train_paths_right, _ = \
        train_test_split(descriptor.steering, descriptor.center,
                         descriptor.left, descriptor.right, test_size=FLAGS.val_portion)
    train_paths = np.concatenate((train_paths_left, train_paths_center, train_paths_right))
    train_steering = np.concatenate((train_steering + FLAGS.shift, train_steering, train_steering - FLAGS.shift))
else:
    train_steering, val_steering, train_paths, val_paths = train_test_split(descriptor.steering, descriptor.center, test_size=FLAGS.val_portion)

train_paths, val_paths = create_paths_to_images(train_paths, FLAGS.data_location), create_paths_to_images(val_paths, FLAGS.data_location)

"""
import matplotlib.pyplot as plt
n, bins, patches = plt.hist(descriptor.steering, bins=n_bins, color='grey')
plt.xlabel('Classes')
plt.ylabel('Samples')
plt.show()
"""

Y_train_binned = continuous_to_bins(train_steering, n_bins=FLAGS.bins)
binned_indices = rebalanced_set(Y_train_binned)

train_paths, train_steering = ensure_valid_values(train_paths, train_steering)
val_paths, val_steering = ensure_valid_values(val_paths, val_steering)

train_paths, train_steering = train_paths[binned_indices], train_steering[binned_indices]
print("Training set size: {}, Validation set size: {}".format(len(train_paths), len(val_steering)))


assert FLAGS.model_type in ['SimplifiedModel', 'SimplifiedModelExtraDropout'], \
    "Incorrect model name provided. Expected: ['SimplifiedModel', 'SimplifiedModelExtraDropout']. Provided {}".\
        format(FLAGS.model_type)
if FLAGS.model_type == 'SimplifiedModel':
    model = SimplifiedModel(FLAGS, input_img_shape)
if FLAGS.model_type == 'SimplifiedModelExtraDropout':
    model = SimplifiedModelExtraDropout(FLAGS, input_img_shape)


config = K.tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

model.compile(loss='mse', optimizer=Adam(1e-2), metrics=['accuracy'])
model.summary() #prints a summary representation of your model.
model_config = model.get_config()
#model = Sequential.from_config(model_config)


model.fit_generator(generate_data_with_augmentation_from(train_paths, train_steering, batch_size, FLAGS.flip),
                    samples_per_epoch=len(train_steering),
                    nb_epoch=FLAGS.epochs,
                    validation_data=generate_data_with_augmentation_from(val_paths, val_steering, batch_size, False),
                    nb_val_samples=len(val_steering))

model.save(os.path.join(model_path, model_name))



