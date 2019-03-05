import pickle
import random

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class PreprocessingDataSet:

    def __init__(self):
        random.seed(1725)
        self.word_features = {}
        self.stop_words = stopwords.words('english')

    # for removing stopping word
    def remove_stopping_word(self, raw_data):
        filtered_list = []
        for w in raw_data:
            if not w in self.stop_words:
                filtered_list = [w]
        return filtered_list

    # have to implement this function to get more accuracy in result
    def getUniqueItems(self, items):
        uniquelist = []
        for item in items:
            if item not in uniquelist:
                uniquelist.append(item)
        return uniquelist

    def word_to_vector(self, word_features):
        return {token: idx for idx, token in enumerate(set(word_features))}

    def find_features(self, document):
        words = word_tokenize(document)
        features = {}
        for w in words:
            features[w] = (w in self.word_features.keys())
        return features

    def get_training_testing_dataset(self, training_percentage=0.5):
        spathneg = "dataset/negativesenwosc.txt"
        spathpos = "dataset/positivesenwosc.txt"

        short_pos = open(spathpos, "r", encoding='utf-8').read()
        short_neg = open(spathneg, "r", encoding='utf-8').read()

        # move this up here
        all_words = []
        documents = []

        # word tag and its meaning  J is for adjective,
        # V is for verb
        allowed_word_types = ["J", "V"]
        print("\n70 30 verb adverb adjective")

        for p in short_pos.split('\n\n\n'):
            documents.append((p, "pos"))
            words = word_tokenize(p)
            pos = nltk.pos_tag(words)
            for w in pos:
                if w[1][0] in allowed_word_types:
                    all_words.append(w[0].lower())

        for p in short_neg.split('\n\n\n'):
            documents.append((p, "neg"))
            words = word_tokenize(p)
            pos = nltk.pos_tag(words)
            for w in pos:
                if w[1][0] in allowed_word_types:
                    all_words.append(w[0].lower())

        # saving documents "no need can be removed"
        save_documents = open("pickle_data_model_save/documents.pickle", "wb")
        pickle.dump(documents, save_documents)
        save_documents.close()

        # counting freq of words
        all_words = nltk.FreqDist(all_words)

        # selecting 5000 word for training
        self.word_features = list(all_words.keys())[:5000]

        # converting word to vector
        self.word_features = self.word_to_vector(self.word_features)

        # saving word_features for testing purpose
        save_word_features = open("pickle_data_model_save/word_features.pickle", "wb")
        pickle.dump(self.word_features, save_word_features)
        save_word_features.close()

        # prepating feature set with tag for training
        featuresets = [(self.find_features(rev), category) for (rev, category) in documents]

        # random shuffling for data set trainig for better accuracy
        random.shuffle(featuresets)
        length = len(featuresets)

        # preparing training and testing dataset for 70-30 percentage
        training_set = featuresets[:int(length * training_percentage)]
        testing_set = featuresets[int(length * training_percentage):]

        return (training_set, testing_set)
