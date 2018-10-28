import argparse
import csv
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sklearn import utils
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Conv2D, Dropout, Flatten, Dense, LeakyReLU
from keras.callbacks import EarlyStopping, TensorBoard

BATCH_SIZE = 32
EPOCHS = 10

def get_model():
    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(100, 200, 3)))
    model.add(Cropping2D(cropping=((34, 0), (0, 0))))
    model.add(Conv2D(24, 5, strides=2))
    model.add(LeakyReLU())
    model.add(Conv2D(36, 5, strides=2))
    model.add(LeakyReLU())
    model.add(Conv2D(48, 5, strides=2))
    model.add(LeakyReLU())
    model.add(Conv2D(64, 3))
    model.add(LeakyReLU())
    model.add(Conv2D(64, 3))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img = cv2.resize(img, (200, 100))
    return img

def generator(samples):
    num_samples = len(samples)
    while 1:
        utils.shuffle(samples)
        for offset in range(0, num_samples, BATCH_SIZE):
            batch_samples = samples[offset:offset + BATCH_SIZE]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for pos in range(3):
                    img_path = Path(batch_sample[pos])
                    img = preprocess(cv2.imread(img_path.as_posix()))
                    images.append(img)

                center_angle = float(batch_sample[3])
                correction = 0.2
                angles.extend([
                    center_angle,
                    center_angle + correction,
                    center_angle - correction
                ])

            for i in range(len(images)):
                images.append(cv2.flip(images[i], 1))
                angles.append(-angles[i])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield utils.shuffle(X_train, y_train)

def plot_history(history):
    epochs = len(history.history['loss'])
    plt.plot(range(1, epochs+1), history.history['loss'], marker="o")
    plt.plot(range(1, epochs+1), history.history['val_loss'], marker="o")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim(bottom=0)
    plt.legend(['Training', 'Validation'])
    plt.savefig('figure.png')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='./data')
    parser.add_argument('--out', default='model.h5')
    args = parser.parse_args()


    samples = []
    with Path(args.dir).joinpath('driving_log.csv').open() as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    train_samples, valid_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(train_samples)
    valid_generator = generator(valid_samples)
    
    es_cb = EarlyStopping(monitor='val_loss', verbose=1)
    tb_cb = TensorBoard(log_dir='log', histogram_freq=0)

    model = get_model()
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(train_generator,
                  steps_per_epoch=int(np.ceil(len(train_samples) / BATCH_SIZE)),
                  validation_data=valid_generator,
                  validation_steps=int(np.ceil(len(valid_samples) / BATCH_SIZE)),
                  callbacks=[es_cb, tb_cb], epochs=10, verbose=1)

    plot_history(history)
    model.save(args.out)

if __name__ == '__main__': main()
