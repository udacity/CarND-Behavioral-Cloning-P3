import sklearn
import numpy as np
from sklearn.model_selection import train_test_split


def loadSamples(sample_path):
    '''
    Loads the CSV sample data
    '''
    import csv
    samples = []
    with open(sample_path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples


def extractSamples(samples, data_path):
    '''
    Extracts needed training data and corresponding measurements
    '''
    image_paths = []
    measurements = []
    for line in samples:
        image_paths.append(data_path + '/' + line[0])  # center_image column
        measurements.append(line[3])  # steering column
    return image_paths, measurements


def generator(samples, batch_size=32):
    '''
    Generate shuffled batch samples on the fly
    '''
    import cv2
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            steerings = []
            for image_path, measurement in batch_samples:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                steerings.append(float(measurement))

            # trim image to only see section with road
            X = np.array(images)
            y = np.array(steerings)
            yield sklearn.utils.shuffle(X, y)


def nvidiaCNN(input_shape):
    '''
    Define the Nvidia End-to-End CNN architecture
    '''
    from keras.models import Sequential, Model
    from keras.layers import Lambda, Cropping2D, Convolution2D, Flatten, Dense
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


def main():
    '''
    Main funtion for training Nvidia CNN model
    '''

    ## Set environment parameters
    data_path = './recording'
    raw_image_shape = (160, 320, 3)
    batch_size = 32

    ## Load data and extract needed inputs
    csvData = loadSamples(data_path)
    image_paths, measurements = extractSamples(csvData, data_path)
    samples = list(zip(image_paths, measurements))

    ## Split train/validation datasets and apply generators
    train_samples, validation_samples = sklearn.model_selection.train_test_split(
        samples, test_size=0.2)
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    ## Instantiate nvidiaCNN() and start training
    model = nvidiaCNN(raw_image_shape)
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(
        train_generator,
        samples_per_epoch=len(train_samples),
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples),
        nb_epoch=3,
        verbose=1)
    model.save('model.h5')


if __name__ == '__main__':
    main()