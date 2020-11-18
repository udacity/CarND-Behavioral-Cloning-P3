# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./doc/model.jpg "Model Architecture"
[image2]: ./doc/center.jpg "Center Image"
[image3]: ./doc/preprocessed_images.jpg "Cropped and flipped images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* data_manip.py containing image manipulation functions
* reader.py containing csv and image reading functions
* cfg.py containing configurations such as directories, filenames
* this writeup.md summarizing the results


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Both maps can be driven in both directions. In order to drive in the reverse direction use manual override mode by pressing W or S keys. Turn back, then release the controls. The car will start driving autonomously.

Note that in case of the 2nd map the connection always gets broken between drive.py and the simulator. This can be seen especially in those cases when the car stops. It can be observed in drive.py's console output that its PI controller increases the throttle but the car doesn't move. Sometimes the car just goes straight off from bends due to the very same reason. Even in these cases it can be observed in the console that drive.py sends out the correct steering commands. These never happen on the 1st map.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

All 27500 data iteams are loaded into the memory. With 32 gigabites of RAM it was no problem, but with 16 or less it might be. Loading the data and training takes ca. 1.5 minutes on an average PC (as of 2020).

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model starts with image cropping and normalization (model.py lines 43-44) 

They are followed by 3 convolution layers rangind from 7 * 7 to 3 * 3 kernel sizes, interpersed by max pooling with 4 * 4 kernel sizes and RELUs. The very first convolitional layer uses 2 * 2 stride to decrease the size of the next layer further (model.py lines 45-53)

After flattening the resulting width is 12800. After a Dropout layer of 0.3 other two fully connected layers come with RELUs.

The last layer contains only a single node, that represents the steering angle.

![Model Architecture][image1]


#### 2. Attempts to reduce overfitting in the model

This model is much bigger than neccessary. The evidences are:
* Training accuracy becomes lover than the validation accuracy as soon as the 2nd epoch. This is a clear sign of overtraining.

Possible solutions:
* Feed it with more data
* Use dropout
* Lessen the network size
* Lessen the number of epochs

Because the model drove very well, I decided to lessen the epochs to only 1. It passed my tests. So I did not feel to battle overtraining anymore. I just added the dropout because it was an expectation in the rubric points. (model.py line 55) 


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 62).

No other tuning was required. This is actually my first iteration, and as it worked perfectly, I didn't fine tune it. The main fine tuning that could be applied is the time consuming experimentatin with decreasing its size up to the point its size and training time is minimized.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.

I used center driving on the first map, and race line driving on the second map. I used 2 full round's image data from driving on the 1st map and 2 full round's image data from driving on the 2nd map. I was driving with keyboard. I planned on doing recovery driving, driving in reverse directions but it was not needed. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to have the model analyze the images at every frame, and have a single output node that servers as steering input.

I previously had good experience with a convolutional network, so I thought that might be a good start. Upscaled it to fit the image size and applied bigger than 1 stride and maxpool 4 * 4 layers to decrease the image size as fast as possible and keeping the useful information at the same time.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I added more input images, dropout and changed the number of the epochs to only one.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The initial model proved to be good, so I did not make any more modifications.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:  

![Center image example][image2]

I collected all data in a ./data folder, and each dataset (run) in a subfolder 01, 02, and so on. These is big amount of data, so it is not uploaded to GitHub.

I also collected the left and right camera images, and modified their steering values by 0.4 and -0.4 respectively so that when the center camera sees something similar during playback, it knows that it should steer back.

To have more useful data I flipped the images and used both the original and flipped ones for training.

![Cropped and flipped images][image3]

I also cropped the top and bottom of the images and normalized the pixel values.

The above proved sufficient to drive the 1st map both in forward and reverse direction, even though I did not collect reverse driving data. 

Then I decided to go for the 2nd map. I collected one lap of data in forward and one lap of data in reverse direction. This proved sufficient for the car to drive both maps in both directions.

There was no need to capture recovey images.

My input data ended up consisting of ca. 27500 images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

The car was able to drive on both maps and directions, but the driving was somewhat jerky. The main reason was that I drove by keyboard and my driving was also jerky. I decided to improve my driving data and repeated all runs with keyboard, and retrained the model. The result was not really surprising:
* the autonomous driving became much smoother
* but at the same time it introduced a strong wobbling left to right.

The reason is probably the fact that the modell was not really faced with recovery situations, and did not learn the difference betweeen straight road and bendings. Possible solutions could be collecting recovery driving data. But due to time pressure I decided for now to fall back to the previously used data. 
