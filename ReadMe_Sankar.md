Build a Traffic Sign Recognition Project
---

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture. Validation accuracy should be greater than 93%
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images



[//]: # (Image References)

[image1]: ./examples/TrainDataVisualization.png "Visualization"
[image2]: ./examples/lenet.png "LeNet Architecture"
[image3]: ./examples/TrainVal_LossAccuracy.PNG "Training & Validation Loss/Accuracy"
[image4]: ./examples/Augmentation.PNG "Augmentation Output"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---

Here is a link to the [project code](https://github.com/mulshankar/TrafficSignClassifier_Sankar/blob/master/TrafficSignClassifier_Sankar.ipynb)

**Data Set Summary**

The dataset used for this work is the German Traffic Dataset located here: http://benchmark.ini.rub.de/

Basic summary of the data set is provided below

The images are already stored in a pickle file as train, validation and test sets. There are in total 43 unique traffic signs to be identified.

The images are of pixel size 32x32 and have 3 channels RGB. Below is the summary of sample set shape and sizes. 

Image Shape: (32, 32, 3)

Training Set:   34799 samples
Validation Set: 4410 samples
Test Set:       12630 samples

**Exploratory visualization of the dataset**

Before training the network on the data set, a quick visualization was done via "seaborn" library. A simple distribution plot along with a gaussian kernel density estimate is plotted to get an idea on how many images are available for each label. Shown below is the visualization. 

![alt text][image1]

**Model Architecture - Design & Test**

The base architecture chosen to fit this data was the LeNet architecture shown below. http://yann.lecun.com/exdb/lenet/

![alt text][image2]

An initial run of the training data set on the LeNet architecture resulted in validation accuracy around 88%. 

- As a first step, a simple normalization scheme was implemented to pre-process the images. The normalization routine chosen was:

	X=(X/127.5)-1

- The min and max pixel values are 0 and 255. The above equation would yield a min of -1 and max of +1 with mean of 0. Network training is a lot easier in such data sets

Despite the normalization scheme, validation accuracy did not go above 91%. 

- As a next step, model complexity was increased by increasing the depth of the convolutional networks as well as adding more layers in the network. 
- While this resulted in validation accuracy more than 93%, loss and accuracy function plots revealed the model overfitting the data.
- Loss and accuracy curve when a model is overfitted is shown below. X-axis is # of epochs

![alt text][image3]

- While the base LeNet architecture already has pooling functionality to avoid overfitting, drop-out technique was used to further avoid overfitting
- Drop-outs were placed at different locations (post convolutional layers/fully connected layers) and tested
- While drop-outs helped in avoiding overfitting, they also resulted in validation accuracy less than 93%
- In order to avoid overfitting and have validation accuracy more than 93%, image augmentation techniques were used to generate more network training data

**Image Augmentation:**

Three basic mechanisms were chosen for augmentation:

- Rotation
- Shear
- Translation 

Augmentation techniques mentioned above were successfully used by Yadav as seen here: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9

- image brightness was another knob that could be tweaked for augmentation but was not used as part of this work
- For each training image, a total of 9 additional images were generated using the above 3 augmentation techniques.
- Ideally, augmentation could be biased towards labels that have less number of training images.
- For the sake of simplicity, all images were increased uniformly in this work. 

Summary of the new data set post augmentation is shown below.

Image Shape: (32, 32, 3)

Training Set:   347990 samples
Training Labels:   347990 labels
Validation Set: 4410 samples
Test Set:       12630 samples

A sample of output from the image augmentation technique is shown below

![alt text][image4]


The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x26 	|
| RELU					|												|
| Avg pooling	      	| 2x2 stride, valid padding, outputs 14x14x26	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Avg pooling	      	| 2x2 stride, valid padding, outputs 5x5x16		|
| Drop-out		      	| keep_prob=0.7									| 
| Flatten				| Output=400 (5x5x16)							|
| Fully connected		| Output=120  									|
| Fully connected		| Output=84  									|
| Dropout				| keep_prob=0.8  									|
| Fully connected		| Output=43  									|
| Softmax				| etc.        									|

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


