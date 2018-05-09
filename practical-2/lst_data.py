#  Base, assigning scores according to the Cosine
# similarity between the target and the substitute word
# embeddings, ignoring the context.
# GOLD: scrap.n::discarded item;album;clipping;piece;crumb;fragment;recycling;morsel;collage;shred;odds and ends;rubbish;

import os
import re
import string
import codecs

class LstIterator:
    def __init__(self, filename, category):
        self.filename = filename
        self.category = category

    def __iter__(self):
        with codecs.open(self.filename, 'r') as f:
            for line in f:
                if self.category=="test":
                    line = line.split()
                    yield LstItem(line)
                else:
                    line = re.split('::|, |\;|, |\n', line)
                    yield GoldSubstitutes(line)

class LstItem:
    def __init__(self, sentence):
        self.target_word = sentence[0].split(".")[0]
        self.complete_word = sentence[0]
        self.sentence_id = sentence[1]
        self.target_pos = sentence[2]
        self.tokenized_sentence = sentence[3:]

class GoldSubstitutes:
    def __init__(self, sentence):
        self.target_word = sentence[0].split(".")[0]
        self.complete_word = sentence[0]
        self.substitutes = sentence[1:-1]
        self.full_sentence = sentence


# if __name__ == "__main__":
#     gold = LstIterator("data/lst/lst.gold.candidates", category="substitute")
#
#     for l in gold:
#         print(l.substitutes)
#         break
