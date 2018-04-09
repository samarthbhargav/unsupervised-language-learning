import os
import bz2
import codecs
import pickle as pkl

import numpy as np

def read_embedding_file(file_name, selected_words = None):
    word_index = []
    embeddings = []

    with bz2.BZ2File(file_name, "r") as reader:

        for idx, line in enumerate(reader):
            if idx % 10000 == 0:
                print("Read {} lines".format(idx))
            line = line.strip().split()
            word = line[0].decode('utf-8')
            if selected_words is not None and word not in selected_words:
                continue
            embedding = np.array(list(map(float, line[1:])))
            word_index.append(word)
            embeddings.append(embedding)

    return word_index, embeddings


def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


class WordSimilarityModel:
    def __init__(self, word_index, embeddings, similarity="cosine"):
        self.word_index = dict(zip(word_index, np.arange(len(word_index))))
        self.embeddings = embeddings

        if similarity == "cosine":
            self.similarity = cosine_similarity
        else:
            self.similarity = similarity

    def __getitem__(self, word):
        if word not in self.word_index:
            raise ValueError("Word '{}' doesn't exist in the model".format(word))
        return self.embeddings[self.word_index[word]]

    def most_similar_to_vector(self, vector, n=10, score=False, word=None):
        word_distances = []
        for other_word, idx in self.word_index.items():
            if word and word == other_word:
                continue
            word_distances.append((other_word, self.similarity(vector, self[other_word])))
        word_distances.sort(key=lambda _: -_[1])
        if score:
            return word_distances[:n]
        else:
            return list(map(lambda _: _[0], word_distances))[:n]

    def most_similar(self, word, n=10, score=False):
        return self.most_similar_to_vector(self[word], n=n, score=score, word=word)

    def get_word_index_matrix(self):
        matrix = np.zeros((len(self.embeddings), len(self.embeddings[0])))
        words = []
        for word, index in self.word_index.items():
            matrix[index] = self[word]
            words.append(word)
        return words, matrix

def load_model(model_name):
    pkl_file = "./models/{}.pkl".format(model_name)
    if not os.path.exists(pkl_file):
        print("Model doesn't exist, creating it")
        word_index, embeddings = read_embedding_file("./data/{}.bz2".format(model_name))
        sim = WordSimilarityModel(word_index, embeddings)
        pkl.dump(sim, open(pkl_file, "wb"))
        return sim
    return pkl.load(open(pkl_file, "rb"))
