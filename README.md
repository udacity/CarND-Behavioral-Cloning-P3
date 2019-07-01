# **Behavioral Cloning** 

TODO: Complete writeup

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

[model.py]: ./model.py
[drive.py]: ./drive.py
[model.h5]: ./model.h5

---
### Foreword

I used pipenv for training the model and executing the simulator on my local machine.

I imitated environment.yml in [CarND-Term1-Starter-Kit](https://github.com/udacity/CarND-Term1-Starter-Kit) provided by Udacity, so I think the environmental difference between my local machine and Udacity workspace is small.

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* README.md summarizing the results (you're reading it)
* [preprocess.py](preprocess.py) for preprocessing image data used by both model training and autonomous driving
* [model.py](model.py) containing the script to create and train the model
* [drive.py](drive.py) for driving the car in autonomous mode
* [model.h5](model.h5) containing a trained convolution neural network 
* [video.mp4](video.mp4) containing a driving video in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py](model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model definition is located in [model.py](model.py) lines 124-165.
The model consists of a convolutional neural network.

The convolutional layers are 3x3 filter sizes and their depths are between 32 and 512 ([model.py](model.py) lines 131-146), and max pooling layers are used.
After that, there are fully connected layers which output sizes are between 1024 and 64.
The model uses RELU layers for activation to introduce nonlinearity.

The image data is preprocessed by the function `preprocess()` ([preprocess.py](preprocess.py)) before input into the network.
Preprocessing consists of cropping, resizing, standardization.
Thus, I inserted preprocess code into [drive.py](drive.py) line 66.


#### 2. Attempts to reduce overfitting in the model

The model contains batch normalization layers in order to reduce overfitting ([model.py](model.py) lines 133, 137, 153).
Batch normalization layers are inserted after each convolutional layer and fully connected layer.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 57).
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually ([model.py](model.py) line 163).


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
I used a center driving data created by myself and project provided dataset.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model based on several convolution layers and max pooling.
It was a basic model I used many time for solving the comupter vision tasks, so I chose it for a starting point.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
To combat the overfitting, I used batch normalization layer implemented in keras.
Batch normalization and dropout techniques cannot be used together, so I used only batch normalization.

However, I realized it was too complicated for me to accomplish all preprocessing tasks I wanted on keras lambda layers.
Thus, I decided to use the plain function to apply preprocessing to images.
This approach also reduced the memory usage of GPU because the model was fed with cropped and resized images.

The final step was to run the simulator to see how well the car was driving around track one.
The vehicle is able to drive autonomously around the track without leaving the road.
However, there were a few spots where the vehicle nearly fell off the track.

1. curved load after the zone with red and white stripe (around 2:40)
2. before the bridge river/lake coin sight (around 1:25)


#### 2. Final Model Architecture

The final model architecture ([model.py](model.py) lines 128-169) consisted of a convolution neural network with the following layers and layer sizes.




#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model.
The validation set helped determine if the model was over or under fitting.

I used an adam optimizer so that manually training the learning rate wasn't necessary.
