import argparse, os
import cv2
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, MaxPooling2D, Lambda
from keras.preprocessing.image import img_to_array, load_img

# Default config
DATA_PATH = 'custom-data/'
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 64, 64, 3
SHAVE_TOP, SHAVE_BOTTOM = 55, 25
TRAIN_VAL_RATIO = 0.8
BATCH_SIZE = 32
EPOCHS, SAMPLES_PER_EPOCH, VAL_SAMPLES = 3, 20000, 3000

#
# --- PRE-PROCESSING ---
#
def randomize_image_brightness(image):
    # Use HSV colour space so we can easily change brightness
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Apply random brightness reduction to V channel.
    # Add constant to prevent complete black images
    random_bright = np.random.uniform() + .25
    image[:, :, 2] = image[:, :, 2] * random_bright

    # Convert back to RGB
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

def crop_image(image):
    roi_top = SHAVE_TOP
    roi_bottom = image.shape[0] - SHAVE_BOTTOM
    cropped = image[roi_top: roi_bottom, :, :]
    return cropped

def resize_image_to_target(image):
    return cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))

def normalize_image(image):
    norm_image = image.astype(np.float32)
    norm_image = norm_image/255.0 - 0.5
    return norm_image

def preprocess_image(image):
    # Input size: 160x320x3, Output size: 64x64x3
    image = crop_image(image)
    image = resize_image_to_target(image)
    image = normalize_image(image)
    return image

#
# --- DATA AUGMENTATION ---
# 
def apply_augmentation(row):
    steering = row['steering']

    # Choose input image from a random camera
    camera = np.random.choice(['center', 'left', 'right'])

    # Adjust the steering angle for left and right cameras
    if camera == 'left':
        steering += 0.25
    elif camera == 'right':
        steering -= 0.25

    image = load_img(DATA_PATH + row[camera].strip())
    image = img_to_array(image)

    # Randomly flip camera image horizontally
    # To reduce bias in the training data for turning left
    flip_prob = np.random.random()
    if flip_prob > 0.5:
        # We need to reverse steering angle when flipping
        steering = -1 * steering
        image = cv2.flip(image, 1)

    # Randomize image brightness
    image = randomize_image_brightness(image)

    # Crop, resize and normalize the image
    image = preprocess_image(image)

    return image, steering

#
# --- TRAINING/VALIDATION DATA GENERATION ---
# 
def get_train_val_data(file_path, train_val_ratio):
    df = pd.read_csv(file_path, usecols=[0, 1, 2, 3])

    # Shuffle data
    df = df.sample(frac=1).reset_index(drop=True)

    # Split training and validation data by a certain factor
    num_train = int(df.shape[0] * train_val_ratio)

    training_data = df.loc[0: num_train-1]
    validation_data = df.loc[num_train:]

    return training_data, validation_data

def get_data_generator(df):
    # Generate data on-the-fly
    total_size = df.shape[0]
    batches_per_epoch = total_size // BATCH_SIZE

    cur_batch = 0

    while(True):
        start = cur_batch * BATCH_SIZE
        end = start + BATCH_SIZE - 1

        image_batch = np.zeros((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), dtype=np.float32)
        steer_batch = np.zeros((BATCH_SIZE,), dtype=np.float32)

        cur_row = 0

        # Take a batch of BATCH_SIZE from dataframe, for each row apply random augmentation on the fly
        for _, row in df.loc[start: end].iterrows():
            image_batch[cur_row], steer_batch[cur_row] = apply_augmentation(row)
            cur_row += 1

        cur_batch += 1
        if cur_batch == batches_per_epoch - 1:
            # Reset batch index when we reach the last batch. 
            cur_batch = 0

        yield image_batch, steer_batch


# --- Convolutional network ---
# 
def get_custom_model():
    model = Sequential()

    # Layer 1. Output shape: 32x32x32
    model.add(Convolution2D(32, 5, 5, 
                            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), 
                            subsample=(2, 2), border_mode="same", name='layer1'))
    model.add(ELU())

    # Layer 2: Output shape: 15x15x16
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid", name='layer2'))
    model.add(ELU())
    model.add(Dropout(.5))
    model.add(MaxPooling2D((2, 2), border_mode='valid'))

    # Layer 3: Output shape: 12x12x16
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid", name='layer3'))
    model.add(ELU())
    model.add(Dropout(.4))

    # Flatten output
    model.add(Flatten())

    # Layer 4
    model.add(Dense(1024, name='layer4'))
    model.add(Dropout(.3))
    model.add(ELU())

    # Layer 5
    model.add(Dense(512, name='layer5'))
    model.add(ELU())

    # Layer 6: Single Output for Steering
    model.add(Dense(1, name='layer6'))

    # Finalize model
    model.compile(optimizer="adam", loss="mse")

    return model

def get_commaai_model():
    model = Sequential()

    # Layer 1
    model.add(Convolution2D(16, 8, 8, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), 
                            subsample=(4, 4), border_mode="same", name='layer1'))
    model.add(ELU())
    
    # Layer 2
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", name='layer2'))
    model.add(ELU())

    # Layer 3
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same", name='layer3'))
    model.add(Flatten())
    model.add(Dropout(.2))

    # Layer 4
    model.add(ELU())
    model.add(Dense(512, name='layer4'))
    model.add(Dropout(.5))

    # Layer 5
    model.add(ELU())
    model.add(Dense(1, name='layer5'))

    # Finalize model
    model.compile(optimizer="adam", loss="mse")

    return model

# --- PROGRAM EXECUTION ---

def init_arg_parser():
    # Example: python model.py -i 'custom-data' -o 'custom-trained-models' -e 3 -w 'sagun' -m 'model1'
    parser.add_argument('-i', '--inputpath', type=str, default=DATA_PATH, 
                        help='Path to driving_log.csv and IMG directory containing JPG files.')
    parser.add_argument('-o', '--outputpath', type=str, default="trained-models", 
                        help='Model output path.')
    parser.add_argument('-m', '--modelname', type=str, default="model", 
                        help='Output model name.')
    parser.add_argument('-e', '--epochs', type=int, default=EPOCHS, choices=range(1, 10), 
                        help='Number of epochs.')
    parser.add_argument('-w', '--whichmodel', type=str, default='sagun', choices=['sagun', 'commaai'], 
                        help='Which model to train on?')
    args = parser.parse_args()

    if os.path.exists('./' + args.inputpath):
        if not args.inputpath[-1:] == '/':
            args.inputpath = args.inputpath + '/'
    else:
        raise ValueError('Input path not found.')

    if not os.path.exists('./' + args.outputpath):
        print('Creating output path: {}...'.format(args.outputpath))
        os.makedirs('./'+args.outputpath)

    if not args.outputpath[-1:] == '/':
        args.outputpath = args.outputpath + '/'

    return args

def save_model(model_path, single_out_file = False):
    if single_out_file:
        # New specs
        model_file = model_path + '.h5'
        if os.path.isfile(model_file):
            os.remove(model_file)
        model.save(model_file)
    else:
        # h5 file
        model_file = model_path + '.h5'
        if os.path.isfile(model_file):
            os.remove(model_file)
        model.save_weights(model_file)
        # json file
        json_file = model_path + '.json'
        if os.path.isfile(json_file):
            os.remove(json_file)
        json_string = model.to_json()
        with open(json_file, 'w' ) as f:
            json.dump(json_string, f)

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Self-Driving Model Training')
    args = init_arg_parser()
    DATA_PATH = args.inputpath

    print("Preparing datasets for training...")
    training_data, validation_data = get_train_val_data(args.inputpath + 'driving_log.csv', TRAIN_VAL_RATIO)
    training_generator = get_data_generator(training_data)
    validation_data_generator = get_data_generator(validation_data)

    print("Initializing {} model...".format(args.whichmodel))
    model = get_custom_model() if (args.whichmodel == 'sagun') else get_commaai_model()

    samples_per_epoch = (SAMPLES_PER_EPOCH // BATCH_SIZE) * BATCH_SIZE

    print("Training convolutional model...")
    model.fit_generator(training_generator, 
                        validation_data=validation_data_generator, 
                        samples_per_epoch=samples_per_epoch, 
                        nb_epoch=args.epochs, 
                        nb_val_samples=VAL_SAMPLES)

    print("Saved model weights and config to {} file".format(args.outputpath + args.modelname))
    save_model(args.outputpath + args.modelname)
    del model