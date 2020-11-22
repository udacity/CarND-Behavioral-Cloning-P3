"""
Improvement ideas:
- ignore X% of 0 steering items
- Use recovery videos
- Gather smoother steering input (mouse, wheel?)
- Gauss smoothing of the road surface
- CLAHE
- HSL: emphasize colors
- nVidia algorithm
- Add throttle / brake
"""

import os
import cfg
import reader
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPool2D, Activation, Dropout, Cropping2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math


if cfg.use_gpu:
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def build_model():
    model = Sequential()
    model.add(Cropping2D(((30, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(filters=16, kernel_size=(7,7), strides=(2,2), padding='same'))  # 160x320x16
    model.add(MaxPool2D((4,4)))  # 40x80x16
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(MaxPool2D((4,4)))  # 10*20*64
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(MaxPool2D((2,2)))  # 5*10*256
    model.add(Activation('relu'))
    model.add(Flatten())  # 12800
    model.add(Dropout(0.3))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.summary(line_length=160)
    return model


def main(first_dataset, last_dataset, verbose=False):
    meta_db = reader.get_all_meta(first_dataset, last_dataset, check_files_exist=True)
    train_meta, valid_meta = train_test_split(meta_db, test_size=cfg.test_size, shuffle=True)
    train_generator = reader.generator(train_meta, batch_size=cfg.batch_size)
    validation_generator = reader.generator(valid_meta, batch_size=cfg.batch_size)
    # X_train, y_train = preprocess(X_train, y_train)

    if verbose:
        pass # show_example(X_train, y_train, start_index=0, columns=3, second_row_offset=len(X_train) // 2)

    datagen_mult = cfg.datagen_item_multiplier if cfg.enable_datagen else 1.0

    model = build_model()
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=cfg.generator_new_item_multiplier * datagen_mult * math.ceil(len(train_meta) // cfg.batch_size),
        validation_data=validation_generator,
        validation_steps=cfg.generator_new_item_multiplier *  datagen_mult * math.ceil(len(valid_meta) // cfg.batch_size),
        epochs=cfg.epochs,
        verbose=1
    )
    model.save(cfg.model_rel_path)


if __name__ == '__main__':
    main(first_dataset=cfg.first_dataset, last_dataset=cfg.last_dataset + 1, verbose=False)

