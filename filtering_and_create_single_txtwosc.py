import re
import time
from os import listdir
from os.path import isfile, join

import memory_profiler 

# Path for testing and traning
trainpathneg = "dataset/aclImdb/train/neg/"
trainpathpos = "dataset/aclImdb/train/pos/"

testpathneg = "dataset/aclImdb/test/neg/"
testpathpos = "dataset/aclImdb/test/pos/"

spathneg = "dataset/negativesenwosc.txt"
spathpos = "dataset/positivesenwosc.txt"


positiveFiles=[]
for x in (trainpathpos, testpathpos):
    for f in listdir(trainpathpos):
        if isfile(join(trainpathpos, f)):
            positiveFiles.append(trainpathpos + f)

negativeFiles=[]
for x in (trainpathneg, testpathneg):
    for f in listdir(trainpathneg):
        if isfile(join(trainpathneg, f)):
            negativeFiles.append(trainpathneg + f)


# class for cleaning and spelling correction
class CleanAndPreprocess:
    def __init__(self):
        self.REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
        self.REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    # function for Cleanind and Preprocessing
    def cleaningSymbol(self, reviews):
        reviews = [self.REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
        reviews = [self.REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
        return reviews


# creating object of class CleanAndPreprocess
camdp = CleanAndPreprocess()


# function for filtering and merging into single file
def filterAndMergePositive():
    wr = open(spathpos, "a", encoding='utf-8')
    for pf in positiveFiles:
        with open(pf, "r", encoding='utf-8') as f:
            line = f.readline()
            line = "".join(camdp.cleaningSymbol(line))
            wr.write(line)
            wr.write("\n\n\n")

    wr.close()
    print('Positive files finished')


def filterAndMergeNegative():
    wf = open(spathneg, "a", encoding='utf-8')
    for nf in negativeFiles:
        with open(nf, "r", encoding='utf-8') as f:
            line = f.readline()
            line = "".join(camdp.cleaningSymbol(line))
            wf.write(line)
            wf.write("\n\n\n")
    wf.close()
    print('Negative files finished')


# running
print('Memory (Before): {}Mb'.format(memory_profiler.memory_usage()))
start = time.time()
filterAndMergePositive()
end = time.time()
print(end - start)

print('Memory (After): {}Mb'.format(memory_profiler.memory_usage()))

start = time.time()
filterAndMergeNegative()
end = time.time()
print(end - start)

print('Memory (After): {}Mb'.format(memory_profiler.memory_usage()))
