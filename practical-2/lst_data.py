import os
import nltk
import string
import codecs

# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english') + list(string.punctuation)

class LstIterator:
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with codecs.open(self.filename, 'r') as f:
            for line in f:
                line = line.split()
                yield LstItem(line)

class LstItem:
    def __init__(self, sentence):
        self.target_word = sentence[0].split(".")[0]
        self.target_postag = sentence[0].split(".")[1]
        self.sentence_id = sentence[1]
        self.target_position = sentence[2]
        self.tokenized_sentence = []
        dummy = sentence[3:]
        for word in dummy:
            if word not in stop_words and word.isalpha():
                self.tokenized_sentence.append(word)

if __name__=='__main__':
    file = LstIterator('data/lst/lst_test.preprocessed')

    for s in file:
        print(s.target_word, s.tokenized_sentence)
    # break
