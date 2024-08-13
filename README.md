# NLP Disaster Tweet Kaggle Project

## Problem Description

The NLP With Disaster Tweets Kaggle competition aims to use NLP specific tools to predict disaster tweets. When using Twitter (or any other social media for that matter), people like to dramaticize what they are saying. The goal of this binary classification project is to distinguish tweets that are about real disasters versus tweets that are not.

For example, take the tweet: "On the plus side, LOOK AT THE SKY LAST NIGHT IT WAS ABLAZE". It is clear to you and I that the author of this tweet was speaking metaphorically, however, the distinction for a computer is much harder. The goal is to build a model that can accurately classify disaster related tweets. 

The accuracy metric for this competition is the F1 score. 

## Data Description

The data supplied contains two files, a train.csv and a test.csv. The train.csv file contains the text of a tweet, a keyword from that tweet, the location that tweet was sent from, and a validation column that predicts whether or not the given tweet is about a real disaster (indicated by a 1) or not (indicated by a 0).

The test.csv file contains a tweet and the label for that particular tweet is what needs to be predicted.

## Data Cleaning

With our data and necessary packages loaded, we can now begin cleaning the data. To clean the data, I will change all the tweets to lowercase and will remove all puncuation. I will also remove all 'stop words' that have been defined by the nltk package. These are words that do not provide any additional insight, think of words like 'a', 'the', 'is', 'are'. By doing this, it will remove noise from the input to our model and can hopefully result in a higher accuracy. 

Let's see how this works with an example.

## Data Visualization

In this next section I will build a few basic visualizations to show the distribution of the target variable and a histogram of the lengths of the individual tweets. 

By doing this exploratory data analysis, both the creator and the viewer of the notebook will gain a better understanding of the structure and characteristics of the data that will be modeled later on in the notebook.

## Data Preprocessing

In the data preprocessing stage, we will continue to prepare our dataset for the modeling phase later in this notebook. Before we can begin modeling, we will need to tokenize the dataset by using the Tokenize package from Tensorflow.

Tokenizing is an NLP pre-processing tool that takes a corpus of text as an input, breaks the document down into individua words, counts the individual words, and finally outputs numerical features that can be used in numerous machine learning models. The Vectorize class is very important, and allows for the model to carry out it's predictive power.

## Model Building

For the first model, we will built a bi-directional LSTM model that will have the ability to classify the tweets as an actual disaster or non-disaster. The model will consist of an embedding layer, two bi-directional LSTM layers, and a dense output layer. The output layer will have a sigmoid activation function.

## Model Training

With our model created, we can now fit the model to the train data that we split earlier in the notebook. The number of epochs will be set to 10 and the batch size to 16. The accuracy will be measured against the validation data.

These parameters were determined by balancing the time it takes for the model to train, and the increased epoch accuracy.

## Model Evaluation

After about 20 minutes of allowing our model to train. We received a max accuracy score of ~83%. However, if you remember, the Kaggle competition stated the accuracy metric is the F1 score. We can use the f1_score package from the sklearn library to easily calculate the score using our predictions.

The overall F1 validation score is ~.724 which is quite good for our initial model. The graph below showcases the validation accuracy and loss over the 10 epochs. Based on the model accuracy graph, it looks like both accuracy scores improved steadily throughout the epochs. The learning rate seemed to be correctly set at .00005. One change that could be made is to increase the number of epochs. Since the accuracy increased steadily throughout each epoch, it would be interesting to increase the number of epochs to see if the accuracy levels off or continues to increase.

## Creating the Submission File

After trainining, fitting, and evaluating the model, we will now use it to make our final predictions on the testing dataset. Then, the predictions will be output to a csv for us to upload to Kaggle.

## Conclusion

In conclusion, throughout this notebook, there were multiple steps that needed to be completed to get a prediction on whether or not the tweet should be classified as an actual disaster or not. 

1. Data Loading and Preprocessing:
    - Loaded both the training and test datasets from Kaggle.
    - Preprocessing was completed by converting the Tweet text to lowercase, removinng non characters, and eliminated sklearn stop words.

2. EDA:
    - The distribution of the target was mostly balanced, with a few more non-disaster Tweets included in the dataset compared to disaster Tweets.
    - The distibution of tweets was centered around 120-140 characters, which helped in padding our model.

3. Model Building and Training: 
    - The Bi-directional LSTM model was created using an embdding layer, two bi-directional LSTM layers, and a dropout layer to prevent overfitting.
    - The model was compliled using an Adam optimizer and included binary cross-entropy as the loss function.
    - The model was trained for 10 epochs and included a batch size of 16.

4. Model Evaluation:
    - The model was evaluated based on the F1 score on the validation dataset, which topped out at a score of .72

In summary, this notebook demonstrated an entire end to end approach to classify text data using a Bi-directional LSTM model. The model performed quite well and with future enhancements, the model could achieve an even better overall accuracy.

### Performance and Potential Improvements

The Bi-directional LSTM model performed well in classifying the disaster tweets. This is clear from the increase in accuracy throughout our epochs. Even with our final max F1 score of .72, there were some areas that could be improved to achieve a higher score:

    - Hyperparameter tuning: Although some hyperparameter tuning was done in the initial model, even more tuning could be done to increase the models predictive power.
    - Additional models: Using other models like Transformer-based architectures could enchance the model's ability.
    - Data Augumentation: Data augumentation such as adding additional relevant Tweets could improve the models predictive ability.

### Future Work

Some future work that could be completed based on the initial approach presented in this notebook could include:
    - Implementing transformer-based models
    - Attempting text pre-processing techinques and data augumentation to allow for additional data to be included
    - Attempting cross validation to ensure the model parameters are optimized.
