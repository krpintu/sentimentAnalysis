# sentimentAnalysis
sentiment analysis


# Sentiment Analysis on Movie Reviews
Introduction
Build an end-to-end sentiment classification system from scratch. The system accepts a movie review as input and classifies it as either positive or negative. There are three main steps:

# Preprocess the data -
Split the data into train and test sets, cleaning dataset, tokenize, stem words, create bag-of-words features, etc.
Models - Create and experiment with different models: Naïve Bayes classification algorithm, Linear Model algorithm, SVM algorithm, etc 
Evaluation - Compare the performances of the models and outline steps to make the chosen model do better.

# Code
sentiment_analysis_test.py - Main code for sentiment analysis testing.
trainingDataSet.py- for training the model

# Setup
Python 3.6+
nltk
sklearn


# Data
Data files for the sentiment analysis project are included under data/imdb-reviews. These are movie reviews from the website imdb.com, each labeled as either 'positive', if the reviewer enjoyed the film, or 'negative' otherwise.

Some of the NLP libraries require additional data for performing tasks like stopwords, PoS tagging, lemmatization, etc. Specifically, nltk will throw an error if the required data is not installed. You can use the following Python statement (in Linux terminal or in a code editor) to open the NLTK downloader and select the desired package(s) to install:

nltk.download("all")

# ###
fill free to improve this code
