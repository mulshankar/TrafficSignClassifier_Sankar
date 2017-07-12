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
[image5]: ./examples/TrainVal_LossAccuracy2.png "Training & Validation Loss/Accuracy Final"
[image6]: ./examples/InternetImages.PNG "Traffic Signs from Google Image Search"
[image7]: ./examples/SoftmaxPlot.PNG "SoftmaxPlot"
[image8]: ./examples/Probabilities_Labels.PNG "Probabilities"

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

- Image brightness was another knob that could be tweaked for augmentation but was not used as part of this work
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
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x36 	|
| RELU					|												|
| Avg pooling	      	| 2x2 stride, valid padding, outputs 14x14x36	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Avg pooling	      	| 2x2 stride, valid padding, outputs 5x5x16		|
| Drop-out		      	| keep_prob=0.7									| 
| Flatten				| Output=400 (5x5x16)							|
| Fully connected		| Output=120  									|
| Fully connected		| Output=84  									|
| Dropout				| keep_prob=0.8  								|
| Fully connected		| Output=43  									|
| Softmax				| etc.        									|

**Model Training** 

- The network parameters used for the architecture are as follows:
	BATCH_SIZE=250
	LEARNING_RATE=0.001
	EPOCHS=7
	OPTIMIZER=Adam
- Using above parameters coupled with architecture described in the table, the loss and accuracy curves obtained in shown in image below

![alt text][image5]

- A distinct difference could be noticed between loss and accuracy curves between above result vs image shown before when model was overfitting. The loss function is a nice monotonically decreasing curve as opposed to a jagged curve
- Also the training and validation accuracies are almost equal with training accuracies even lower than validation accuracies indicating that the model is not overfitting
- With validation accuracy about 95%, the model was tested on the test set
- Test accuracy with the model was about 91%

**Model validation on new images obtained from Google**

- While the above experiment was performed on the german traffic sign data set, it will be very interesting to evaluate model performance on new images obtained from just a simple google image search
- The 5 images downloaded were sanitized to a format similar to training data i.e 32x32 pixels with R,G and B channels
- The 5 traffic signs chosen for this test is shown below

![alt text][image6]

- Other than bumpy road traffic sign, which has a slight orientation and some blue blob next to it, other signs are expected to be detected fairly easy 
- Accuracy on the internet images was 0.6 => 3/5 images were detected accurately
- The top 5 softmax probabilities were sorted using the tf.nn.top_k function and shown below comparing to true labels

![alt text][image8]

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Entry      		| No Entry   									| 
| Pedestrians  			| General Caution								|
| Turn Right			| Round About Mandatory							|
| Bumpy Road      		| Bumpy Road					 				|
| Speed Limit 80		| Speed Limit 80      							|


- Interestingly, the model identified No Entry, Bumpy Road and Speed Limit 80 km/hr signs accurately
- Significant conclusions cannot be drawn from testing on just 5 images
- It is interesting to note that the model confidence on predictions was fairly high despite the loss and accuracy curves showing no overfitting of the model

![alt text][image7]


