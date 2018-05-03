import numpy as np
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
