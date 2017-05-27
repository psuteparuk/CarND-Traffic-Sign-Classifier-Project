# Traffic Sign Recognition

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[sign-example]: ./writeup-images/sign-example.png "Sign Example"
[training-distribution]: ./writeup-images/class-distribution-training.png "Class Distribution in Training Data"
[validation-distribution]: ./writeup-images/class-distribution-validation.png "Class Distribution in Validation Data"
[before-pre-processed]: ./writeup-images/before-pre-processed.png "Original image (before pre-processing)"
[after-pre-processed]: ./writeup-images/after-pre-processed.png "After pre-processing"
[before-augmented]: ./writeup-images/before-augmented.png "Original image (before augmented)"
[augmented-images]: ./writeup-images/augmented-images.png "Augmented images"
[new-test-images]: ./writeup-images/new-test-images.png "New test images"
[new-test-prediction]: ./writeup-images/new-test-prediction.png "Prediction on the new test images"
[top-5]: ./writeup-images/top-5.png "Top 5 Softmax Probabilities"

## Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it! and here is a link to my [project code](https://github.com/psuteparuk/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used the Numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is 32x32x3. (The 3 at the end is the number of color channels.)
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an example of a traffic sign in our data set. Each sign is a 32x32 colored image.

![alt text][sign-example]

We also look at the class distribution for the training and validation data. From the bar charts below we can see that some classes are well represented while some are rarely seen. This could affect our model since it will be bias toward the better represented class. However, we are not very concern with this problem as we believe the distribution reflects the real-world distribution of traffic signs. The distributions from both the training and the validation data set are also very similar so this skewed distribution should not impact our validation accuracy.

![alt text][training-distribution]

![alt text][validation-distribution]

The class with the fewest examples has 180 examples in the training data, while the class with the highest number of examples has 2010 examples.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.

As a first step, I applied the histogram equalization to the each image to take away the brightness effect. Brightness plays no part in the characteristic of the traffic sign and impedes a classification model. So we should be able to safely remove them from our calculation.

The next step I took was normalization by the formula `(image - 127.5) / 255`. This is to ensure that the input is all within the same range of value. In this case, it is `[-0.5, 0.5]`. I applied this to every set of data (training, validation, test).

Here is an example of a traffic sign image before and after the histogram equalization and normalization.

Original data:

![alt text][before-pre-processed]

After Pre-processed:

![alt text][after-pre-processed]

At first this is the only pre-processing I did. However, after training some models on the pre-processed data, I soon realized that the network could not identify some classes very well since there are very few of them in the training data (some classes have less than 200 images). In addition, the model seems to have a high variance problem. This prompts me to consider adding more data to the training examples.

To add more data to the data set, I used the techniques inspired by [Vivek Yadav](https://github.com/vxy10/ImageAugmentation/blob/master/img_transform_NB.ipynb). In particular, I applied some small random affine transformation (translation, rotation, and shearing) and brightness to each original image. This is a very cheap way to get more data compared to going out to get some images from the actual signs. The jittering is small enough such that the network could still roughly identify that it is of the same class as the original.

Here is an example of an original image and an augmented image:

Original data:

![alt text][before-augmented]

Augmented data:

![alt text][augmented-images]

Note that we are keeping the classes distribution the same. As stated above, I believe the given distribution reflects the real-world traffic sign distribution, so it is fine to be bias. However, the augmented data should help under-represented classes to have more data.

Originally, we have 34799 training data. I augmented the data 4x so we now have 139196 training data.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.)

My final model consisted of the following layers:

| Layer                                                 | Description                                  |
|-------------------------------------------------------|----------------------------------------------|
| Input                                                 | 32x32x3 RGB image                            |
| 5x5 Convolution with 32 filters (activated with ReLU) | 1x1 stride, valid padding, outputs 28x28x32  |
| Inception Module (activated with ReLU)                | See GoogLeNet paper, outputs 28x28x128       |
| Max pooling                                           | 2x2 stride, valid padding, outputs 14x14x128 |
| Dropout                                               | 0.5 keep probability                         |
| Inception Module (activated with ReLU)                | See GoogLeNet paper, outputs 14x14x256       |
| Max pooling                                           | 2x2 stride, valid padding, outputs 7x7x256   |
| Dropout                                               | 0.5 keep probability                         |
| Fully Connected (activated with ReLU)                 | outputs 256 neurons                          |
| Fully Connected (activated with ReLU)                 | outputs 43 neurons (number of sign classes)  |

At the heart of this network is the inception module introduced by the GoogLeNet [paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf). My network consists of two such modules with each one followed immediately by a max pooling layer and a dropout with 0.5 keep probability.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I calculated the logits from the above network. I then calculated the cross entropy between the softmax of those logits and the one-hot encoded of class labels. This is the loss of my model.

I trained the model with 15 epochs and a batch size of 128. From observation the accuracy seems to plateau after the 10th epoch. For extension beyond this project, I might experiment with early stopping when the accuracy change is insignificant to speed up the process.

I used the Adam optimizer with a fixed learning rate of 0.001 to minimize this loss. I tried different learning rate values from 0.0001 to 0.01 and for this particular model, the 0.001 rate seems to give the highest validation accuracy. I also did try out the exponential decay learning rate but the accuracy improvement is insignificant.

I also add an L2 regularization to alleviate the overfitting. I used 0.001 as the coefficient for the regulariation term.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 98.3%
* test set accuracy of 97.4%

The process is iterative. I started out with the LeNet-5 network with 10 epochs and 0.001 learning rate. This gave me a 89.3% validation accuracy. After adding the pre-processing and experimenting with the number of epochs and the learning rate value, it occurred to me that the validation accuracy was actually worse. For a certain setting, I got 73.7% validation accuracy while the training accuracy was at 96.1%. This suggested that the network is highly overfit.

To mitigate this problem, I introduced the L2 regularization term to penalize large weights and biases. After some experiments, the LeNet-5 with 10 epochs, 0.005 learning rate and a regularization coefficient of 0.001 gave me 94.3% validation accuracy and 98.6% training accuracy.

I have seen that the accuracy tends to go up and down over later epochs. So I tried to lower the learning rate by introducing the exponential decay. But the change is insignificant so I dropped it.

Through various settings, the validation accuracy seemed to plateau at around 94-95% even after regularization. The training accuracy also plateau at 98%. With a bias problem, I augmented the data 4x. The accuracy improved but still plateaued around 96%. I decided to try out a new architecture. GoogLeNet is well-known for its success in the ImageNet competition and is also very fast. So I decided to use it as a new base. The inception module is also great at combining coarse features and detailed features together and let the network choose which one is best at each layer.

Using two inception modules, I could get the validation accuracy up to 97.6% with 15 epochs and a learning rate of 0.001. At this point, the training accuracy has hit 99%. To alleviate the slight overfit, other than adding the regularization, I also added a max pool layer and a dropout after each inception modules. Finally, I could get the validation accuracy to 98.3%.

One caveat is that the training time was around 20 minutes for each epoch. I decided to train on the augmented data only for the first five epochs and then on the original training data for the rest. The augmented data provide noisy data so the network can first learn coarse features. I suspected that removing the augmented data after a few epochs should not hurt the performance and it went as I suspected. We gained some speed without compromising the accuracy.

The final test accuracy is at 97.4%.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 12 more German traffic signs that I found on the web (courtesy of [NikolasEnt](https://github.com/NikolasEnt/Traffic-Sign-Classifier/tree/master/new_images):

![alt text][new-test-images]

Here I introduce some new images such as the 40km/h sign and the No U Turn sign. They are not part of the 43 classes of sign in our data set, but it is interesting to see what our model would do.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Here are the results of the prediction:

![alt text][new-test-prediction]

The model was able to correctly guess 8 of the 12 traffic signs. The 40km/h sign is expected to be an incorrect prediction since the network has not encountered it before. However, it did realize that the sign is a speed limit. As for the No U Turn sign, the network best guess is General Caution, which is somewhat close. If we take out these two examples, our network performs at a 80% accuracy.

The other incorrect predictions are all from speed limit signs. The model does realize that they are speed limits but incorrectly identify the numbers. For future improvement, we could incorporate a specialized OCR classifier into our model when we classify a sign as a speed limit.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

Here is the top 5 softmax probabilities on each image:

![alt text][top-5]

We can see that for the two unknown cases, the model is very unsure about its prediction with the highest probabilities only in the range of 25%-35%. For others, the prediction seems to be very clear cut. The two incorrect speed limits predictions show that the network knows they are speed limits, but could not read the numbers correctly.
