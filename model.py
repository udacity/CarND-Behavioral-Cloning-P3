import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def read_file(filename):
    """
    reads the file using csv library and returns rows in the file
    """
    lines = []
    with open(filename) as csvfile:
        data_rows = csv.reader(csvfile)
        for row in data_rows:
            lines.append(row)
    return lines

def crop_images(X, y):
    """
    This method calculates the top and bottom percentages and crops the image
    Resulting shape is (72, 320, 3)
    No. of Output Images = No. of Input Images
    """
    images = []
    steering_angles = []
    top_percent = 0.4
    bottom_percent = 0.15
    
    for i in range(len(X)):
        ind_img = X[i]
        top = int(np.ceil(ind_img.shape[0] * top_percent))
        bottom = ind_img.shape[0] - int(np.ceil(ind_img.shape[0] * bottom_percent))
        cropped_img = ind_img[top:bottom, :]
        images.append(cropped_img)
        steering_angles.append(y[i])
    return images, steering_angles

#Without resizing gave better results, hence don't use this
def resize_images(X, y):
    """
    This method resizes the images to height=66, widht=200
    No. of Output Images = No. of Input Images
    """
    images = []
    steering_angles = []
    for i in range(len(X)):
        resized = cv2.resize(X[i], (200, 66))
        images.append(resized)
        steering_angles.append(y[i])
    return images, steering_angles

    
def apply_gamma(X, y):
    """
    This method applies gamma filter to the input images
    Observe the gamma images are added to the original data set
    No. of Output Images = 2 * (No. of Input Images)
    """
    images = []
    steering_angles = []
    for i in range(len(X)):
        gamma = np.random.uniform(0.7, 1.7)
        inv_gamma = 1 / gamma
        map_table = np.array([((i/255.0)**inv_gamma)*255 for i in np.arange(0,256)])
        transformed_img = cv2.LUT(X[i], map_table)
        images.append(X[i])
        steering_angles.append(y[i])
        images.append(transformed_img)
        steering_angles.append(y[i])
    return images, steering_angles

def vary_brightness(X, y):
    """
    This method alters the brightness of the image by a random value
    uses HSV color space as V represents brightness
    No. of Output Images = No. of Input Images
    """
    images = []
    steering_angles = []
    for i in range(len(X)):
        # HSV (Hue, Saturation, Value) - Value is brightness
        hsv_img = cv2.cvtColor(X[i], cv2.COLOR_RGB2HSV)
        random_value = 1.0 + 0.6 * (np.random.rand() - 0.5)
        hsv_img[:,:,2] =  hsv_img[:,:,2] * random_value
        transformed_img =  cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        images.append(transformed_img)
        steering_angles.append(y[i])
    return images, steering_angles

    
def flip_images_and_add(X, y):
    """
    This method flips the input images
    Flips are done only for those images where steering angles are outside the range of (-0.1, +0,1)
    This means straight or near straight steering angle images are not flipped as it doens't add any value
    No. of Output Images > No. of Input Images
    """
    #print('size before', len(X))
    images = []
    steering_angles = []
    for i in range(len(X)):
        #print('less or greater {}'.format(y[i]))
        images.append(X[i])
        steering_angles.append(y[i])
        #Flip only those images where there are curves
        if y[i] < -0.1 or y[i] > 0.1 :
            images.append(cv2.flip(X[i], 1))
            steering_angles.append(y[i] * -1.0)
    return images, steering_angles

def translate(X, y, range_x, range_y):
    """
    This method randomly translates the image in any direction 
    and calculates the corresponding change in the steering angle
    """
    images = []
    steering_angles = []
    for i in range(len(X)):
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        transformed_angle = y[i] + trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = X[i].shape[:2]
        transformed_img = cv2.warpAffine(X[i], trans_m, (width, height))
        images.append(X[i])
        steering_angles.append(y[i])
        images.append(transformed_img)
        steering_angles.append(transformed_angle)
    return images, steering_angles

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def data_generator(rows, validation_flag, batch_size):
    """
    This is the Python Generator that reads values in chunks
    and makes it possible to run in modest CPUs
    """
    correction_factor = 0.20
    path = 'trainingdata/IMG/'
    len_rows = len(rows)
    rows = shuffle(rows)
    while 1:
        for offset in range(0, len_rows, batch_size):
            batch_rows = rows[offset:offset+batch_size]
            images = []
            steering_values = []
            #print('rows in batch', len(batch_rows))
            for line in batch_rows:
                
                center_image_path = line[0]
                left_image_path = line[1]
                right_image_path = line[2]

                center_image_name = center_image_path.split('/')[-1] #Last token [-1] is the image
                left_image_name = left_image_path.split('/')[-1]
                right_image_name = right_image_path.split('/')[-1]

                center_image_bgr = cv2.imread(path+center_image_name)
                left_image_bgr   = cv2.imread(path+left_image_name)
                right_image_bgr = cv2.imread(path+right_image_name)
                
                #Converting from BGR to RGB space as simulator reads RGB space
                center_image = cv2.cvtColor(center_image_bgr, cv2.COLOR_BGR2RGB)
                left_image   = cv2.cvtColor(left_image_bgr, cv2.COLOR_BGR2RGB)
                right_image = cv2.cvtColor(right_image_bgr, cv2.COLOR_BGR2RGB)
                
                

                steering_value = float(line[3])
                left_steering_value = steering_value + correction_factor
                right_steering_value = steering_value - correction_factor
                
                images.append(cv2.GaussianBlur(center_image, (3, 3), 0))
#                 images.append(center_image)
                steering_values.append(steering_value)

                images.append(cv2.GaussianBlur(left_image, (3, 3), 0))
#                 images.append(left_image)
                steering_values.append(left_steering_value)
                
                images.append(cv2.GaussianBlur(right_image, (3, 3), 0))
#                 images.append(right_image)
                steering_values.append(right_steering_value)
                
            
            X_train, y_train = images, steering_values
            X_train, y_train = shuffle(X_train, y_train)

            #Augmenting & Pre-processing
            #X_train, y_train = crop_images(X_train, y_train)
            #X_train, y_train = resize_images(X_train, y_train)
            X_train, y_train = translate(X_train, y_train, 100, 10)
            X_train, y_train = flip_images_and_add(X_train, y_train)
            X_train, y_train = vary_brightness(X_train, y_train)
            X_train, y_train = shuffle(X_train, y_train)

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            yield X_train, y_train

        

from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Reshape
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D

#Architecture based on NVIDIA
def train_model(train_generator, valid_generator, len_train, len_valid):
    """
    This method contains the definition of the model
    It also calls methods to train and validate the data set
    """
    
    print('Training started...')

    model = Sequential()
    #model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(72, 320, 3)))
    model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    #model.add(Reshape((55, 135)))
    model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    
    
    start_time = time.time()
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch= len_train, validation_data=valid_generator, nb_val_samples=len_valid, nb_epoch=10)
    print('Training complete!')
    print('Total time for training {:.3f}'.format(time.time() - start_time))
    model.save('model.h5')

    

def mainfn():
    """
    This is the main function that kicks-off the process
    """
    data_rows = read_file('./trainingdata/driving_log.csv')
    print('Length of the csv file {}'.format(len(data_rows)))
    
    rows_train, rows_valid = train_test_split(data_rows, test_size=0.2)
    #print('splitting done {} {}'.format(len(rows_train), len(rows_valid)))
    
    train_generator = data_generator(rows_train, False, batch_size = 32)
    valid_generator = data_generator(rows_valid, True, batch_size = 32)
    #print('generator invoked train {} valid {}'.format(train_generator, valid_generator))
    
    train_model(train_generator, valid_generator, len(rows_train), len(rows_valid))

#Calling the mainfn() to kick-off the process
mainfn()