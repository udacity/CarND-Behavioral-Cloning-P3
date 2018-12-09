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


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./center_2018_12_08_16_16_09_662.jpg "Grayscaling"
[image3]: ./right_2018_12_08_16_16_09_942.jpg "Recovery Image"
[image4]: ./right_2018_12_08_16_16_10_496.jpg "Recovery Image"
[image5]: ./right_2018_12_08_16_16_10_844.jpg "Recovery Image"
[image6]: ./center_2016_12_01_13_31_12_937.jpg "Normal Image"
[image7]: ./center_2016_12_01_13_31_12_937_flipped.jpg "Flipped Image"
[image8]: ./cnn-architecture-624x890.png "DAVE-2 Image"
[image9]: ./mse_graph.png "Without dropout"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network ( Generated with my local machine )
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model used NVIDIA's DAVE-2 CNNs ( shown below )
![DAVE-2][image8]

NVIDIA's intention is to input YUV image, but I used it with RGB.
Model is described between line 111 - 126 in model.py.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 121, 123, 125 ).

The model was trained and validated on different data sets to ensure that the model was not overfitting.
 - From different recordings (code line 41).

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 128).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
I used a combination of center , left image and right image.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

- Implemented LeNet based CNN.
  It worked correctly and can control almost appropriately, but it goes off the road if I run it through night ( 12 Hour long run ).
- Implemented DAVE-2 CNN.
- In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 
- To combat the overfitting, I added three dropout layers. 
    ![over fitting ][image9]
- At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 111-126) consisted of a convolution neural network with the following layers and layer sizes translated from NVIDIA DAVE-2 CNN.
            _________________________________________________________________
            Layer (type)                 Output Shape              Param #   
            =================================================================
            lambda_4 (Lambda)            (None, 160, 320, 3)       0         
            _________________________________________________________________
            cropping2d_4 (Cropping2D)    (None, 65, 320, 3)        0         
            _________________________________________________________________
            conv2d_16 (Conv2D)           (None, 31, 158, 24)       1824      
            _________________________________________________________________
            conv2d_17 (Conv2D)           (None, 14, 77, 36)        21636     
            _________________________________________________________________
            conv2d_18 (Conv2D)           (None, 5, 37, 48)         43248     
            _________________________________________________________________
            conv2d_19 (Conv2D)           (None, 3, 35, 64)         27712     
            _________________________________________________________________
            conv2d_20 (Conv2D)           (None, 1, 33, 64)         36928     
            _________________________________________________________________
            flatten_4 (Flatten)          (None, 2112)              0         
            _________________________________________________________________
            dense_13 (Dense)             (None, 100)               211300    
            _________________________________________________________________
            dropout_10 (Dropout)         (None, 100)               0         
            _________________________________________________________________
            dense_14 (Dense)             (None, 50)                5050      
            _________________________________________________________________
            dropout_11 (Dropout)         (None, 50)                0         
            _________________________________________________________________
            dense_15 (Dense)             (None, 10)                510       
            _________________________________________________________________
            dropout_12 (Dropout)         (None, 10)                0         
            _________________________________________________________________
            dense_16 (Dense)             (None, 1)                 11        
            =================================================================
            Total params: 348,219
            Trainable params: 348,219
            Non-trainable params: 0

Here is a visualization of the DAVE-2 architecture from Web
![DAVE-2][image8]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![Almost off the course][image3]
![Started recovery][image4]
![Recovered][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help reduce deviation of data set.
For example, here is an image that has then been flipped:

![Normal image][image6]
![Flipped image][image7]


After the collection process, I had 8106 number of data points. I then preprocessed this data by normalize around 0 center.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was Z as evidenced by ...
I used an adam optimizer so that manually training the learning rate wasn't necessary.
