import pickle
import time
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier 
from Preprocessing import PreprocessingDataSet

pre = PreprocessingDataSet()

training_set, testing_set = pre.get_training_testing_dataset(training_percentage=0.5)
print(len(training_set), len(testing_set))



start=time.clock()

# training in all model to check better accuracy and saving model for future perdiction
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)

save_classifier = open("pickle_data_model_save/originalnaivebayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

print('time taken by NaiveBayesClassifier ',(time.clock()-start))



start=time.clock()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

save_classifier = open("pickle_data_model_save/MNB_classifier.pickle", "wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

print('time taken by MultinomialNB ',(time.clock()-start))



start=time.clock()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

save_classifier = open("pickle_data_model_save/BernoulliNB_classifier.pickle", "wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

print('time taken by BernoulliNB ',(time.clock()-start))




start=time.clock()

# in LogisticRegression_classifier we use hyperparameter C, which adjusts the regularization.
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:",
      (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

save_classifier = open("pickle_data_model_save/LogisticRegression_classifier.pickle", "wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

print('time taken by LogisticRegression ',(time.clock()-start))




start=time.clock()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

save_classifier = open("pickle_data_model_save/LinearSVC_classifier.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

print('time taken by LinearSVC ',(time.clock()-start))



start=time.clock()

SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDClassifier accuracy percent:", nltk.classify.accuracy(SGDC_classifier, testing_set) * 100)

save_classifier = open("pickle_data_model_save/SGDC_classifier.pickle", "wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()

print('time taken by SGDClassifier ',(time.clock()-start))




start=time.clock()

decisionTreeClassifier = SklearnClassifier(DecisionTreeClassifier())
decisionTreeClassifier.train(training_set)
print("DecisionTreeClassifier accuracy percent:", nltk.classify.accuracy(decisionTreeClassifier, testing_set) * 100)

save_classifier = open("pickle_data_model_save/DecisionTreeClassifier.pickle", "wb")
pickle.dump(decisionTreeClassifier, save_classifier)
save_classifier.close()

print('time taken by DecisionTreeClassifier ',(time.clock()-start))
