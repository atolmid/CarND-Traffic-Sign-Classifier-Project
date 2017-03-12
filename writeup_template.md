#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/data_exploration.png "Visualization"
[image2]: ./examples/random_initial_image.png "Random Sign"
[image3]: ./examples/gaussian.png "Gaussian Filter"
[image4]: ./examples/laplace.png "Laplace Filter"
[image5]: ./Images/0.jpg "Traffic Sign 1"
[image6]: ./Images/1.jpg "Traffic Sign 2"
[image7]: ./Images/2.jpg "Traffic Sign 3"
[image8]: ./Images/3.jpg "Traffic Sign 4"
[image9]: ./Images/4.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/atolmid/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
In addition, there is a second version of the code, where there is no data augmentation:
[project code v2](https://github.com/atolmid/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier%20-%20Copy.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the union of the training and test set labels, in order to get  the number of classes.
I derived the set of the union (which has unique elements), and its length provides the number of unique classes.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed between the different classes.
This is shown for both the training (blue colour), as well as the test set (red colour).
It is clear from this graph, that the dataset is unbalanced.
Labels such as "1", or "2" appear much more frequently than "42" for example.

![alt text][image1]

The fourth code cell contains the code that plots the image of a random sign from the training dataset.

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth up to eleventh code cells of the IPython notebook.

As a first step, I decided to convert the images to grayscale to check whether this would help increase the classification accuracy.
Since the results appeared not to be better than using coloured images (in most experiments they were worse), the specific code was commented out, and the rest was done using the coloured images.


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

A validation set was already provided, thus there was no need to split the training dataset.
The training set was augmented as previously described, in order to generate more data, and improve the accuracy of the predictions of the trained model.

My final training set had 104397 number of images. My validation set and test set had 4410 and 12630 number of images.

In the sixth code cell, I used the scipy.ndimage library, in order to generate more data.
I used two filters for data augmentation:
* gaussian_filter
* laplace

Here is an example of the same traffic sign image, using the gaussian, and then the Laplace filter.

![alt text][image3]
![alt text][image4]

Then, I combined the new datasets with the original, and deleted the temporary datasets.
Finally, I normalised all data (training, validation, test), by dividing all pixel values by 255.0

I normalized the image data because this way the values of the weights in the neural networks would not become too big during training.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the twelfth cell of the ipython notebook. 

My final model was building on the LeNet network, and consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Fully connected		| Input = 400. Output = 120      				|
| RELU					|												|
| Dropout				|           									|
| Fully connected		| Input = 120. Output = 84      				|
|       				|        									|
|						|												|
|						|												|
 
For the cross entropy I used softmax\_cross\_entropy\_with\_logits.
In addition, I used l2 regularisation on the weights and biases of the last two layers (tf.nn.l2\_loss()).
This was selected for the same reason dropout was used, to reduce overfitting.


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the cells 13-17 of the ipython notebook. 

The optimiser used was Adam, with a learning rate of 0.001.
A learning rate of 0.0001 was also used initially, however the model needed significantly more epochs to train, so the value of 0.001 was the one eventually used,
The beta for the l2 regularisation was also set to 0.001

To train the model, I used a batch size of 128, and in the case where data augmentation was used, I let it train for 500 epochs (in the version with no data augmentation, the epochs were 200).


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the sixteenth cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 94.5% (in the case without data augmentation the accuracy was 95.6%) 
* test set accuracy of 92.3% (in the case without data augmentation the accuracy was 93.2%) 


* Initially the LeNet-5 architecture was selected, as suggested in the exercise instructions
* The problem was that the accuracy was below 90%
* Then, initially dropout was added, and the accuracy was around 93% (not always above that threshold)
* The next step was to add the l2 regularisation. As already mentioned, it was used, in the same sense as dropout, to reduce overfitting. This way, the accuracy reached the level of 95%.
* Finally, data augmentation was used. However in the specific experiments conducted, it did not bring the expected improvement in accuracy

* Since the accuracy on the test set is not significantly lower than that of the validation set, it is rather safe to say, that the model does not have a serious overfitting problem.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

The first and the third images should be relatively easy to predict, as they appear often in the dataset.
the other three might be difficult to classify because their frequency in the dataset is not so high, and they might be mistaken with other signs that are similar, but appear more often.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the twenty-third cell of the Ipython notebook.

Here are the results of the prediction with data augmentation:

| Image			                     |     Prediction	        					| 
|:----------------------------------:|:--------------------------------------------:| 
| Speed limit (30km/h)               | Speed limit (60km/h)   				        | 
| Road narrows on the right          | General caution 							    |
| Road work				             | Speed limit (30km/h)							|
| Pedestrians	      	             | General caution				 				|
| End of all speed and passing limits| End of no passing      					    |

Here are the results of the prediction without data augmentation:

| Image			                     |     Prediction	        					| 
|:----------------------------------:|:--------------------------------------------:| 
| Speed limit (30km/h)               | Priority road        				        | 
| Road narrows on the right          | Road narrows on the right 				    |
| Road work				             | Right-of-way at the next intersection		|
| Pedestrians	      	             | General caution					 			|
| End of all speed and passing limits| End of speed limit (80km/h)     			    |

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the twenty-third cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (60km/h) sign (probability of 0.69), and the image does contain a Speed limit sign, however (30km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .69         			| Speed limit (60km/h)  						| 
| .30     				| Speed limit (80km/h) 							|
| 1.2e-03 				| End of speed limit (80km/h)					|
| 1.4e-05	      		| Speed limit (50km/h)			 				|
| 1.3e-07			    | Speed limit (30km/h)    						|

For the second image, the model is very sure that this is a General caution sign (probability of 0.99), however the image contains a Road narrows on the right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| General caution 					        	| 
| 5.7e-03        		| Traffic signals 								|
| 2.5e-07   			| Speed limit (30km/h)							|
| 9.0e-08	    		| Pedestrians					 				|
| 2.1e-08			    | Speed limit (20km/h)      					|

For the third image, the model is relatively sure that this is a Speed limit (30km/h) sign (probability of 0.67), however the image contains a Road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .67         			| Speed limit (30km/h) 						    | 
| .33     				| Right-of-way at the next intersection 		|
| 2.6e-05   			| Speed limit (20km/h)							|
| 2.0e-05	   			| Speed limit (100km/h)							|
| 9.3e-06			    | Pedestrians      			        			|

For the fourth image, the model is relatively sure that this is a General caution sign (probability of 0.41), the image contains a Pedestrians sign though. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .41         			| General caution 						        | 
| .39     				| Road work	 									|
| .09					| Traffic signals								|
| .04	      			| Dangerous curve to the left					|
| .04				    | Pedestrians     					    		|

For the fifth image, the model is very sure that this is a End of no passing sign (probability of 0.98), but the image contains a End of all speed and passing limits sign. The top five soft max probabilities were

| Probability         	    |     Prediction	        					    | 
|:-------------------------:|:-------------------------------------------------:| 
| .98         			    | End of no passing  					   	        | 
| .01     				    | End of all speed and passing limits    			|
| 5.6e-07					| End of no passing by vehicles over 3.5 metric tons|
| 1.5e-07	      			| End of speed limit (80km/h)					 	|
| 6.6e-13				    | Children crossing      							|

As can be seen, even though in several cases the correct prediction was in the top 5 (3 out of 5 times), the model failed to pick the correct sign as the top prediction.
This can be because of with the resizing the new images were a bit distorted, or for some of them that they did not appear often in the training data set, and the model had a bias against them.

One solution could be to select a balanced dataset out of the training set, and train the model with that.

The version without the data augmentation manages to successfully predict the second sign (Road narrows on the right) with 0.99997 probability.
The reader can refer to the relevant linked jupyter notebook for more details, as well as a graphical representation of the softmax probabilities, and the top-5 predictions for each case.

###Visualize the Neural Network's State with Test Images

There was a problem getting the layer weights.
Due to lack of time, I did not insist so much on the optional part.