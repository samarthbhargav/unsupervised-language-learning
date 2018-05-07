import os
import string
import codecs

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
        self.complete_word = sentence[0]
        self.sentence_id = sentence[1]
        self.target_position = sentence[2]
        self.tokenized_sentence = sentence[3:]
