# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is as bellow.

| Layer           | Description
|:----------------|:-----------
| Input           | 3@100x200 YUV image
| Normalization   |
| Cropping        | 3@66x200
| Convolution 5x5 | 24@30x98 (stride: 2x2; padding: VALID; activation: ReLU)
| Convolution 5x5 | 36@14x47 (stride: 2x2; padding: VALID; activation: ReLU)
| Convolution 5x5 | 48@5x22  (stride: 2x2; padding: VALID; activation: ReLU)
| Convolution 3x3 | 64@3x20  (stride: 1x1; padding: VALID; activation: ReLU)
| Convolution 3x3 | 64@1x18  (stride: 1x1; padding: VALID; activation: ReLU)
| Dropout         | keep_prob: 0.5
| Flatten         | 1164
| Fully-connected | 100
| Fully-connected | 50
| Fully-connected | 10
| Fully-connected | 1

#### 2. Attempts to reduce overfitting in the model

- The model contains dropout layers in order to reduce overfitting ([model.py lines 28](https://github.com/eduidl/CarND-Behavioral-Cloning-P3/blob/master/model.py#L28)).

- The model was trained and validated on different data sets to ensure that the model was not overfitting ([model.py lines 98-108](https://github.com/eduidl/CarND-Behavioral-Cloning-P3/blob/master/model.py#L98-L108)).
- The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually ([model.py lines 103](https://github.com/eduidl/CarND-Behavioral-Cloning-P3/blob/master/model.py#L103)).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

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


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
