<h1 align="center"> Food Vision101 🍔👁 </h1> 


We're going to be building Food Vision Big™, using all of the data from the Food101 dataset.

All 75,750 training images and 25,250 testing images.

This time we've got the goal of beating [DeepFood](https://www.researchgate.net/publication/304163308_DeepFood_Deep_Learning-Based_Food_Image_Recognition_for_Computer-Aided_Dietary_Assessment), a 2016 paper which used a Convolutional Neural Network trained for 2-3 days to achieve 77.4% top-1 accuracy.

Google Colab offers free GPUs (thank you Google), however, not all of them are compatiable with mixed precision training.

Google Colab offers:

 * <i>K80 (not compatible)</i>
 * <i>P100 (not compatible)</i>
 * <i>Tesla T4 (compatible)</i>

Knowing this, in order to use mixed precision training we need access to a Tesla T4 (from within Google Colab) or if we're using our own hardware, our GPU needs a score of 7.0+ (see here: https://developer.nvidia.com/cuda-gpus).

<H3>1. Problem Definition</H3>

In a statement,

> Predict food image out of 101 category of foods, food image recognition algorithm.

<H3>2. Data</H3>

The original data came from the TensorFlow Datasets (TFDS), read the guide: https://www.tensorflow.org/datasets/overview

There is also a version of it available on Kaggle. https://www.kaggle.com/dansbecker/food-101

<H3>3. Steps</H3>


 * Using TensorFlow Datasets to download and explore data
 * Creating preprocessing function for our data
 * Batching & preparing datasets for modelling (making our datasets run fast)
 * Creating modelling callbacks
 * Setting up mixed precision training
 * Building a feature extraction model
 * Fine-tuning the feature extraction mode

<h3>4. Result</h3>

Trained a computer vision model with competitive performance to a research paper and in far less time (our model took ~60 minutes to train versus DeepFood's quoted 2-3 days) and model beat the results mentioned in the DeepFood paper for Food101 (DeepFood's 77.4% top-1 accuracy versus our ~83.8% top-1 accuracy).

<Br>
If you liked what you saw, want to have a chat with me about the portfolio, work opportunities, or collaboration, shoot an email at <a href="mailto:zeeshanrajpootkr@gmail.com?subject=Hello%20Anmol" target="_top">
zeeshanrajpootkr@gmail.com</a> 
