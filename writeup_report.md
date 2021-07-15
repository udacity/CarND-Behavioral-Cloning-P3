# **Behavioral Cloning** 
In this project, a convolutional neural network is trained to clone a driving behavior from a simulator. The CNN architecture used here is referenced from NVIDIA's [End to End Learning for Self-Drivig Cars paper](https://arxiv.org/pdf/1604.07316v1.pdf).

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

[//]: # (Image References)

[image1]: ./CNNarchitecture.png "CNN architecture"

---
## 1. The Goal of the Project
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---
## 2. Data Collection Strategy
The data is collected via the simulator.

1. I just collected the driving data for a 1-round. 
2. And optionally it would be possible to drive the opposite direction for one turn.  
3. Actually, opposite direction data set can be replaced by data augmentation of previous data flipping. As a result, data was doubled with data augmentation.

---
## 3. Pipeline
The pipeline consist of, six steps as follows: 

### 1. Import the required libraries

### 2. Loading the raw data

### 3. Data Augmentation
- Images are cropped to remove irrelevant parts of the training (the car's hood and the sky)
- Images are flipped and their corresponding steering angle values are multiplied by -1

### 4. Image Pre-Processing
- Image is resized (200x66)

### 5. Creating the CNN Architecture using Keras  
Following CNN architecture is used as shown below:
![alt text][image1]  
The CNN architecture is used from NVIDIA's End to End Learning for Self-Driving Cars paper
> This network architecture consists of 9 layers, including a normalization layer, three 5x5 convolutional layers,
two 3x3 convolution layers and three fully connected layers.  
      
> The below is a model structure output from the Keras which gives more details on the shapes and the number of parameters.  

>
 | Layer (type)                   |Output Shape      |Params  |Connected to     |
 |--------------------------------|------------------|-------:|-----------------|
 |lambda_1 (Lambda)               |(None, 66, 200, 3)|0       |lambda_input_1   |
 |convolution2d_1 (Convolution2D) |(None, 31, 98, 24)|1824    |lambda_1         |
 |convolution2d_2 (Convolution2D) |(None, 14, 47, 36)|21636   |convolution2d_1  |
 |convolution2d_3 (Convolution2D) |(None, 5, 22, 48) |43248   |convolution2d_2  |
 |convolution2d_4 (Convolution2D) |(None, 3, 20, 64) |27712   |convolution2d_3  |
 |convolution2d_5 (Convolution2D) |(None, 1, 18, 64) |36928   |convolution2d_4  |
 |dropout_1 (Dropout)             |(None, 1, 18, 64) |0       |convolution2d_5  |
 |flatten_1 (Flatten)             |(None, 1164)      |0       |dropout_1        |
 |dense_1 (Dense)                 |(None, 100)       |115300  |flatten_1        |
 |dense_2 (Dense)                 |(None, 50)        |5050    |dense_1          |
 |dense_3 (Dense)                 |(None, 10)        |510     |dense_2          |
 |dense_4 (Dense)                 |(None, 1)         |11      |dense_3          |
 |                                |**Total params**  |252219  |                 |

### 6. Compile and Save the Model
- I used Mean Square Error (MSE) loss function to measure how close the model predeicts to the given steering and for each image.
- I used Adam optimizer for optimization with learning rate 1.0e-3(default rate) 
- Finally, the model is compiled and saved.