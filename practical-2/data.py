import codecs
import os
import collections

import numpy as np

def read_stop_words(file_name):
    with codecs.open(file_name, "r", "utf-8") as reader:
        return [line.strip() for line in reader]

class SentenceIterator:

    def __init__(self, file_name, min_length = 0, max_length = 50, stop_words = None):
        self.file_name = file_name
        self.min_length = min_length
        self.max_length = max_length
        self.stop_words = stop_words

    def __iter__(self):
        with codecs.open(self.file_name, "r", "utf-8") as reader:
            for line in reader:
                line = line.split()
                if len(line) < self.min_length or len(line) > self.max_length:
                    continue
                # remove stop_words
                if self.stop_words:
                    line = [word for word in line if word.lower() not in self.stop_words]
                yield line

def get_context_words(sentence, index, context_window):
    n_ = context_window // 2
    context = set()
    for i in range(index - n_, index + n_ + 1):
        if i == index or i < 0 or i >= len(sentence):
            continue
        context.add(sentence[i])
    return context



class Vocabulary:

    # TODO remove stop words
    def __init__(self, sentences, max_size = 10000, special_tokens = None):
        if special_tokens is None:
            special_tokens = {"$UNK$", "$EOS$", "$SOS$", "$PAD"}
        self.index  = {}

        counts = collections.defaultdict(int)
        sentence_count = 0
        for sentence in sentences:
            sentence_count += 1
            for word in sentence:
                counts[word] += 1

        print("Number of sentences: {}. Found a total of {} unique words in the data. Picking the top {}".format(sentence_count, len(counts), max_size))
        counts = list(counts.items())
        counts.sort(key=lambda _: -_[1])
        most_freq = [w[0] for w in counts[: max_size - len(special_tokens)]]

        self.index = dict([ (w, i) for i, w in enumerate(most_freq)])
        for special_token in special_tokens:
            assert special_token not in self.index
            self.index[special_token] = len(self.index)


        self.inverse_index = dict([(v, k) for (k, v) in self.index.items()])

        self.N = len(self.index)

    def one_hot(self, word, strict = False):
        if word not in self.index:
            if strict:
                raise ValueError(" '{}' not present in the vocabulary".format(item)) ;
            word = "$UNK$"
        vector = np.zeros(self.N)
        vector[self.index[word]] = 1
        return vector

    def word(self, index):
        return self.inverse_index[index]

    def __getitem__(self, item):
        if item not in self.index:
            raise ValueError(" '{}' not present in the vocabulary".format(item))
        return self.index[item]

if __name__ == '__main__':
    sentences = SentenceIterator("data/hansards/training.en")
    v = Vocabulary(sentences)
