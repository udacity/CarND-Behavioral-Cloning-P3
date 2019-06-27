import pathlib

import pandas
import cv2
import numpy
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPool2D, BatchNormalization, Activation, Flatten, Dense


#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

def main():
    pwd_path = pathlib.Path(__file__).parent
    #csv_path = pwd_path / "data/driving_log.csv"
    csv_path = pwd_path / "vendor/data/driving_log.csv"
    dataset = load_dataset(csv_path)

    input_shape = (160, 320, 3)
    model = create_model(input_shape)

    model.fit(dataset.X, dataset.y,
            batch_size=32,
            validation_split=0.2,
            shuffle=True,
            epochs=10,
            verbose=1,
            )

    model.save('model.h5')

#------------------------------------------------------------------------------
# Data
#------------------------------------------------------------------------------

class Dataset:
    def __init__(self, X, y):
        self.X = numpy.array(X)
        self.y = numpy.array(y)
        assert(len(self.X) == len(self.y))
        self.size = len(X)

    def __len__(self):
        return self.size

def load_dataset(csv_fname):
    df = pandas.read_csv(csv_fname)

    dir_path = pathlib.Path(csv_fname).parent
    def pick_image(relative_fname):
        img_path = dir_path.joinpath(rel_fname)
        return cv2.imread(str(img_path))
    X = [pick_image(fname) for fname in df["center"].values]

    y = df["steering"].values.astype('float32')

    return Dataset(X, y)


#------------------------------------------------------------------------------
# Model
#------------------------------------------------------------------------------

def create_model(input_shape):
    model = Sequential()

    #model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))

    model.add(Flatten(input_shape=input_shape))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model


if __name__ == "__main__":
    main()
