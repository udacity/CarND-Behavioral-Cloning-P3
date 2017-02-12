
# **Behavioral Cloning** 

---

**Behavioural Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Augmentation_CenterLeftFlipped.png "Augmentation steps: Center, Left, and Flipped"
[image2]: ./examples/RandomBrightness.png "Augment Random brightness"
[image3]: ./examples/CroppedROI.png "Image Cropped to ROI"
[image4]: ./examples/Resized.png "Resized to 64x64"
[image5]: ./examples/DifferentCamera.png "Images from Left, Center and Right Camera"
[image6]: ./examples/SteeringAngleDistribution.png "Histogram showing Steering Angle Distribution"
[image7]: ./examples/SteeringTemporalPlot.png "Plot of Steering versus Time"
[image8]: ./examples/Driving_Centre.png "Training by center lane driving"
[image9]: ./examples/Driving_RecoveryFromLeft.png "Training by recovering from left back to center"
[image10]: ./examples/Driving_RecoveryFromRight.png "Training by recovering from right back to center"
[image11]: ./examples/RandomAugmentedImages.png "Random examples of final augmented and pre-processed images"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.json + model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py trained_models/model_sagun.json
```

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Also, I added argument parser for my own convenience in training my model between different training data, different architectures and epochs. Example usage (see model.py lines 210-236):
```sh
python model.py -i 'udacity-data' -o 'trained-models' -e 5 -w 'sagun' -m 'model_sagun'
python model.py -i 'custom-data' -o 'trained-models' -e 3 -w 'commaai' -m 'model_commaai'
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model (model.py lines 136-174) uses 32 5x5 filters and 16 3x3 in the following two layers. I used 2x2 max pooling. I used two fully connected layers with 1024 neurons and a final layer with one output for steering. The model includes Exponential Linear Unit (ELU) also known as leaky-ReLU layers to introduce nonlinearity and prevents the problem of vanishing gradients which is a drawback of ReLU. Also, it is used to make transition between angles smoother. The data is normalized in the pre-processing pipeline before sending it to the network instead of using lambda layer. The architecture will be discussed further in the later paragraphs. 

I initially started testing Comma-AI's network which trained with very good validation accuracy but with bad results on the actual track. This model is also available in model.py line 176-205.

#### 2. Attempts to reduce overfitting in the model

The model contains three dropout layers of probabilities 0.5, 0.4, 0.3 following conv layers 2, 3, 4 respectively in order to reduce overfitting (model.py lines 148, 154, 161). 

I experimented with various dropout probabilities where I initially used aggressive dropouts. However, since the car did not perform well on the track, the final parameters seemed to work well. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. 


#### 3. Model parameter tuning

The model used an adam optimizer so the learning rate was not tuned manually (model.py line 203) and used mean squared error (MSE) as the loss function.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the Comma-AI model. I thought this model might be appropriate because of its small size and its popularity in the CarND forums.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I spent a lot of time training various models ranging from 1 to 10 epochs. Although I found that more epochs resulted low mean squared error on the training set but higher than 8 epochs, I noticed a high mean squared error on the validation set. So this was a sign of overfitting. So number of epochs used during the later stages was 5.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. My initial guess was that I should not trust my own training data. I used the dataset provided by Udacity, which clearly showed exaggerated amounts of recovery data and the vehicle would oscillate like a drunk driver and always fall off the bridge. 

Another approach I tried was merging my own data with udacity data. The behaviour resulting from this merged data was less oscillating, very slow to train because of more data, yet was not successful in completing the track.

I then decided to try my own model, sightly larger but using dropout and ELU layers as in Comma-AI model. After various trials and tribulations my final model architecture as distribed above got the job done. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. Better yet, the model was able to generalise to Track 2 even when no training data was provided for this track. I would say that the second track performed even better than the first one!

```
20000/20000 [==============================] - 58s - loss: 0.0581 - val_loss: 0.0214
Epoch 2/5
20000/20000 [==============================] - 57s - loss: 0.0454 - val_loss: 0.0041
Epoch 3/5
20000/20000 [==============================] - 57s - loss: 0.0419 - val_loss: 5.4761e-06
Epoch 4/5
20000/20000 [==============================] - 57s - loss: 0.0406 - val_loss: 0.0018
Epoch 5/5
20000/20000 [==============================] - 57s - loss: 0.0409 - val_loss: 0.0020
Saving model weights and config to trained-models/model_sagun file...
```

#### 2. Final Model Architecture

My final model architecture (model.py lines 136-174) consisted of a convolution neural network with the following layers and layer sizes:

1. **Layer 1**: Convolutional layer with 32 5x5 filters, ELU activation
2. **Layer 2**: Convolutional layer with 16 3x3 filters, ELU activation, Dropout(0.5) and 2x2 max pooling
3. **Layer 3**: Convolutional layer with 16 3x3 filters, ELU activation, Dropout(0.4)
4. **Layer 4**: Fully connected layer with 1024 neurons, Dropout(0.3) and ELU activation
5. **Layer 5**: Fully connected layer with 1024 neurons, Dropout(0.3) and ELU activation

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image8]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover back to the center of the lane when it begins to veer off. These images show what a recovery looks like starting from left and right respectively.

![alt text][image9]

![alt text][image10]

For this, I was careful in recording images only when the vehicle is entering the recovery and not veering off so that the network does not begin mimicking unwanted behaviour of veering off. I repeated this process for two laps in Track 1 in order to get more data points. I did not use any training data from Track 2. 

After the collection process, I had 5921 number of data points. 

I plotted up a histogram distribution of steering angles and found that my training data set was heavily biased towards going straight and turning left. This is mainly because of Track 1 in which the training was done which has a lot more left turns than right. 

![alt text][image6]

I also analysed a Steer vs. Time for 500 image frames and found out the steering pattern of my training data.

![alt text][image7]

The simulator provided images from left, center and right camera for each frame recorded as shown below for Frame 30.

![alt text][image5]

As my training dataset was not large enough, I had to augment synthetic data for training the convolutional network. 

I used the techniques suggested in the [Vivek Yadav's blog](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9) and [nVidia's paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), for augmenting data. 

### Augmentation strategy

- Without having to generate tons of data by myself, I would use the left and right camera images to simulate the effect of the car wandering off to the side and recovering. Experimentally, if the left camera is steered by adding a small angle of 0.25 radians (6.25 degrees), it would have similar image as the center camera. Similarly, if 0.25 angle is subtracted from the right camera image, it would be similar to the center camera image. In this way, the camera location was chosen randomly from the dataset and if it was left or right, 0.25 was added or subtracted respectively.

As mentioned previously, there are a lot of images of the car turning left than the right. So to reduce this bias, I flipped the image horizontally (along y=0 axis) randomly. However, when a flip was triggered, the steering angle should be reversed. 

![alt text][image1]

To make the convolutional network more robust to changes in brightness (both day and night), and pre-empting the problem of Track 2, which was darker compared to Track 1, I added random brightness to the training images. The easiest way to do this is to convert the RGB image to HSV colour space and darken it by a random factor. However, to prevent completely black images, a lower limit of 25% darkness was added. After this, the image was converted back to RGB colour space. 

![alt text][image2]

### Pre-processing strategy

The lower part of the camera images contained the hood of the car. Also, the horizon and the sky was not necessary for learning to drive the structured track. Therefore, I removed the top 55 pixels and the bottom 25 pixels which resulted in reducing the size of the image in half from 160x320 to 80x320.

![alt text][image3]

Finally, the image is resized to size 64x64 which is sufficiently small size for the network to learn. 

![alt text][image4]

A number of random images going through the entire augmentation and pre-processing pipeline is shown below.

![alt text][image11]

## Data Generation strategy

I finally randomly shuffled the data set and put Y% of the data into a validation set. 
 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 

## Training strategy

The ideal number of epochs for me was 5 as described earlier. I used an adam optimizer so that manually training the learning rate wasn't necessary.

I used `fit_generator` which is part of Keras to generate random new augmented/pre-processed data on the fly. Since there is a limit in memory and the amount of training data for deep learning is very large, this on the fly would enable the optimizer to see virtually unlimited new data on every epoch. The `fit_generator` called my custom python generator which did the following.
- For training set, I generated 20,000 images on the fly per epoch
- For validation set, I generated 3,000 images on the fly per epoch

I did not use any test set as the model was deemed successful if it drove reasonably well on the complete Track 1 without going out of the line or getting stuck even once.


### Evaluation Video

Initially, I used `video.py` to generate the video but the quality was really poor so I decided to do a screen capture instead. 

Please excuse my taste for the background music in the video but it was getting really boring staring at the same silent simulation environment for weeks. :)

**Successful autonomous driving on both [Track 1](https://www.youtube.com/watch?v=qODMKNyRgpY) and [Track 2](https://www.youtube.com/watch?v=qODMKNyRgpY&t=1m29s)**

[![Autonomous Mode Track 1 + 2](https://img.youtube.com/vi/qODMKNyRgpY/0.jpg)](https://www.youtube.com/watch?v=qODMKNyRgpY)
