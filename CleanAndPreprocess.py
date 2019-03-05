import re

from nltk.stem.snowball import SnowballStemmer


class CleanAndPreProcess:
    def __init__(self):
        self.stemmer=SnowballStemmer("english")
        self.REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
        self.REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    # function for Cleanind and Preprocessing
    def cleaningSymbol(self, reviews):
        reviews = [self.REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
        reviews = [self.REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
        return "".join(reviews)

    def plural_to_singular_doc(self,words):
        temp=[]
        for word in words.split():
            temp.append(self.stemmer.stem(word)+" ")
        return "".join(temp)

    
