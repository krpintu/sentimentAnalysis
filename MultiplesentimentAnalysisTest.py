import random
import pickle
from avgvoteclassifier import AVGVoteClassifier
from nltk.tokenize import word_tokenize

path_pickle_model="pickle_data_model_save"

word_features5k_f = open(path_pickle_model+"/word_features.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


# def find_features(document):
#     words = word_tokenize(document)
#     features = {}
#     for w in word_features.keys():
#         features[word_features[w]] = (w in words)
#     return features

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in words:
        features[w] = (w in word_features.keys())
    return features

open_file = open(path_pickle_model+"/originalnaivebayes.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()

open_file = open(path_pickle_model+"/MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open(path_pickle_model+"/BernoulliNB_classifier.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open(path_pickle_model+"/LogisticRegression_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open(path_pickle_model+"/LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open(path_pickle_model+"/SGDC_classifier.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = AVGVoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

def sentimentSentance(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)
