import codecs
import os
import collections


class SentenceIterator:

    def __init__(self, file_name, min_length = 0, max_length = 50):
        self.file_name = file_name
        self.min_length = min_length
        self.max_length = max_length

    def __iter__(self):
        with codecs.open(self.file_name, "r", "utf-8") as reader:
            for line in reader:
                line = line.split()
                if len(line) < self.min_length or len(line) > self.max_length:
                    continue
                yield line

class Vocabulary:

    def __init__(self, sentences, max_size = 10000, special_tokens = None):
        if special_tokens is None:
            special_tokens = {"$UNK$", "$EOS$", "$SOS$"}
        self.index  = {}

        counts = collections.defaultdict(int)
        for sentence in sentences:
            for word in sentence:
                counts[word] += 1

        print("Found a total of {} unique words in the data. Picking the top {}".format(len(counts), max_size))
        counts = list(counts.items())
        counts.sort(key=lambda _: -_[1])
        most_freq = [w[0] for w in counts[:max_size]]

        self.index = dict([ (w, i) for i, w in enumerate(most_freq)])
        for special_token in special_tokens:
            assert special_token not in self.index
            self.index[special_token] = len(self.index)
            
        self.N = len(self.index)

    def __getitem__(self, item):
        if item not in self.index:
            raise ValueError(" '{}' not present in the vocabulary".format(item))
        return self.index[item]

if __name__ == '__main__':
    sentences = SentenceIterator("data/hansards/training.en")
    v = Vocabulary(sentences)
