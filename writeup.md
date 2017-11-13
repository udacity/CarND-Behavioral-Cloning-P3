# Udacity Car-ND

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_cnn_architecture.png "Model Visualization"
[image2]: ./examples/model_layers_table.PNG "Grayscaling"
[image8]: ./examples/combined_corrected_total.png "Combined After Balance"
[image9]:./examples/hist_of_original_data.png "Combined Before Balance"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py  -- the script to create and train the model
* drive.py  -- for driving the car in autonomous mode
* model.h5 -- a trained convolution neural network
* writeup_report.md -- summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes 'ELU' layers to introduce nonlinearity (code line
216-234), and the data is normalized in the model using a Keras lambda
layer (code line 219).

####2. Attempts to reduce overfitting in the model

The model was split (80/20) to avoid over fitting. Eighty percent of the
data was used as training data, and twenty percent was used as the
validation set. This helps to test if the model is actually learning how
to drive, or just learning the exact input images. If it only learns the
exact imput images, it won't learn what to do if it finds itself on a
new road, or even a different part of the track.
#### 3. Model parameter tuning

The model had 6 hyperparameters I adjusted to try and find the best working model.
* correction = 0.25
  * Images were taken from the left, right, and center cameras. For
    images taken from the left and right camera, the steering angles
    were adjusted by the correction factor in order to simulate more
    data closer to the left and right side of the road.
* epochs = 25
* border_mode = 'same'
* activation = 'elu'
* optimizer = 'adam'
* batch_size = 64
* myData = True
* uData = True
* preprocess = True

There were also two parameters to determine if I wanted to collect the
histograms, and if I wanted them to be "printed" to the screen.
* getHist = True
* Show = True

Finally, I had a hyper parameter to select if I wanted to gather the
data directly from the source file, or to use a pickled version. Loading
the pickles is a much faster process, but doesn't include any new data I
may have recorded.
* Use Pickles = True

#### 4. Appropriate training data

Since this is a Behavioral Cloning project, its necessary to generate
data we want the model to emulate in autonomous mode. We don't want to
input data that represents unsafe driving, such as off the edge of the
road or into the water. We want to give the model examples of center
lane driving, and what to do if it finds itself at or beyond the lane lines.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to
the LeNet architecture. I started with this simply because it was
familiar and I had used it in previous projects. After watching the
walkthrough vidio from Udacity, I decided to utilize the Nvidia
architecture and found it to be a very effective model for behavioral
cloning.

After doing some reading on other models and some of the other writeups
available on GitHub, I saw that a recurrent theme was that "it's all
about the data". So I decided to focus on getting a model that worked
reasonably well, but also to have more than enough data going into the
model that it would be able to function properly. I collecected four
laps of regular driving, two clockwise and two counter clockwise. Then I
collected roughly two laps of "recovery" data where I drove the car to
the edge of the road with the recording off, and then drove it back to
the center with the recording on. This served to train the model what to
do when the car gets to the edge of the road, and proved to be a pretty
effective technique.

Another small technique I found to be helpful was to keep the mouse held
down the whole time I was steering. Rather than letting the car drive
straight (steering_angle = 0) and then correcting, this allowed for
small steering angles and more gradual changes while driving, which
helped to even out the data set, and the training model.

In order to improve areas in which the car had difficulty, such as the
dirt patch or the bridge, I drove those portions multiple times with
both normal driving and recovery driving. It also seemed to help to
preprocess the images from RGB to HSV. This way the color of the dirt
didn't confuse the model as much, since it could still look for features
in the HSV color spectrum.

At the end of the process, the vehicle is able to drive autonomously
around the track without leaving the road. it was a little jerky, and I
would like to come back and do more adjustments to the model when I have
time, but it was very satisfying to see tangible improvements. In the
traffic sign project, changes to the model simply change the validation
accuracy at the end, but seeing a car actually drive correctly was very
rewarding.

#### 2. Final Model Architecture

The final model architecture is as follows:

![alt text][image2]

I based my model architecture off of the Nvidia end to end neural
network found
[here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).
Their model is visualized as:
 ![alt text][image1]

#### 3. Training Laps

After reading some of the commentary from other students, I learned from
them that, "it's all about the data". Because of this, I decided to
collect copious amounts of data. First I drove four laps of center lane
driving where I did my best to keep the car in the center of the road. I
drove two of the laps in the clockwise direction, and two laps in the
counter clockwise direction. This was to teach the model "normal"
driving.

Then I recorded two laps of "recovery" driving where I would drive the
car to the edge of the lane with the recording turned off, and "recover"
with the recording turned on. This way the model would learn what to do
when the car gets off track and finds itself at the edge of the lane.

#### 4. Generating more data

Recording extra laps is only one approach to generating more training
data. Another approach is to augment the images and measurements already
generated. This creates slightly different variants of the input data. To
that end, I used three different methods:

##### a. Flipping
I flipped the images and their measurements, and included those also in
the dataset. This way, the model would learn a more general approach of
how to drive on the track, instead of overfitting to the data available
from training laps. Without flipping, the model tends to only learn how
to turn left, because of the bias created by a left turn bias in the
training track.

##### b. Balancing the Histogram
Most often, the car doesn't need to make sharp turns to get around the
track. Good driving results mostly in pretty low steering angles. This
bias tends to teach the model that driving straight is more likely to be
the correct steering angle, so it tends to just drive straight all the
time. To counteract this, I took images with greater steering angles and
duplicated them. For each of the duplicate images, I randomly altered
the brightness so the model wouldn't overfit to seeing the exact images
over and over again.

Before balancing the dataset, this was the data distribution:
![alt text][image9]

After balancing the dataset, this was the data distribution:
![alt text][image8]


All together, with the center, left, and right images, as well as the
flipped images, the total data for the whole model was 145932 images and
measurements.

#### 5. Random Shuffle
Using the keras model.fit function, the dataset was shuffled, with 20%
siphoned off for validation training.

I used this training data for training the model. The validation set
helped determine if the model was over or under fitting. The ideal
number of epochs was 4. After 4 epochs, the validation loss began to
increase. I used an adam optimizer so that manually training the
learning rate wasn't necessary.

#### 6. Challenges

Most of the challenges I encountered generally had more to do with using
Python, Tensorflow, or Keras than with building the model. A big lesson
of this project was the importance of sanity checking the data. I ran
many iterations of the model with poor results because I didn't realize
that a typo of "images" instead of "myImages" was excluding all of my
images and was assigning the steering angle measurements from my data to
the udacity training data. After this, I made sure to sanity check all
the data on every run. I did this by insuring all the data from my data
and from the udacity data was included, and all the numbers matched up.
Needless to say, this drastically improved results.

I also had diffuculty getting the GPU on my computer to run the gpu
version of tensorflow. There was a steep learning curve learning all
about cuDNN, cuda, and the environment path. Ultimately, the
[SentDex](https://pythonprogramming.net) tutorials helped a lot, and I
was able to get the environment up and running.

