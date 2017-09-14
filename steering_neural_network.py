from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam


class SteeringNeuralNetwork():

    def __init__(self, input_shape, output_shape, curr_model=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        if curr_model:
            self.model = curr_model
        else:
            self.model = self.create_network(input_shape, output_shape)

    @staticmethod
    def create_network(input_shape, output_shape):
        activation = "relu"

        model = Sequential()

        # Normalize
        model.add(BatchNormalization(input_shape=input_shape, axis=1))

        model.add(Conv2D(24, (5, 5), strides=(2, 2), activation=activation, padding='same', name="convolution0"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        model.add(Conv2D(36, (5, 5), strides=(2, 2), activation=activation, padding='same', name="convolution1"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        model.add(Conv2D(48, (5, 5), strides=(2, 2), activation=activation, padding='same', name="convolution2"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        model.add(Conv2D(64, (3, 3), activation=activation, padding='same', name="convolution3"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        model.add(Conv2D(64, (3, 3), activation=activation, padding='same', name="convolution4"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))


        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Activation(activation=activation))

        model.add(Dense(1164, activation=activation, name="dense1"))
        model.add(Dropout(0.5))
        model.add(Activation(activation=activation))

        model.add(Dense(100, activation=activation, name="dense2"))
        model.add(Dropout(0.5))
        model.add(Activation(activation=activation))

        model.add(Dense(50, activation=activation, name="dense3"))
        model.add(Dropout(0.5))
        model.add(Activation(activation=activation))

        model.add(Dense(10, activation=activation, name="dense4"))
        model.add(Dropout(0.5))
        model.add(Activation(activation=activation))

        model.add(Dense(output_shape, name="output"))

        adam = Adam(lr=1e-04)
        model.compile(optimizer=adam, loss="mse")
        return model