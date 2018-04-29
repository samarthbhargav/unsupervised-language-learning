## The Skip-Align model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distb
from torch.autograd import Variable
from data import SentenceIterator, Vocabulary, get_positive_context

from pprint import pprint
from collections import Counter
import numpy as np

class SkipGramModel(nn.Module):
    def __init__(self, vocab, embedding_dim):
        super(SkipGramModel, self).__init__()

        self.vocab = vocab

        self.embedding_dim = embedding_dim

        # TODO add noise
        self.input_embeddings = torch.zeros(self.embedding_dim, vocab.N)
        self.input_embeddings += torch.rand(self.input_embeddings.size())
        self.input_embeddings = Variable(self.input_embeddings, requires_grad=True)

        self.output_embeddings = torch.zeros(vocab.N, self.embedding_dim)
        self.output_embeddings += torch.rand(self.output_embeddings.size())

        self.output_embeddings = Variable(self.output_embeddings, requires_grad=True)

    def forward(self, center_word, positive_words, negative_words):
        input_embedding = torch.matmul(center_word, self.input_embeddings.t()).view(1, -1)

        positive_embeddings = torch.matmul(positive_words, self.input_embeddings.t())
        negative_embeddings = torch.matmul(negative_words, self.output_embeddings.t())

        # print(input_embedding.size())
        # print("Input embedding:", input_embedding.size(), "Positive :", positive_embeddings.size())
        pos_score  = torch.matmul(input_embedding, positive_embeddings.t()).squeeze()
        pos_score = F.logsigmoid(pos_score)
        pos_score = torch.sum(pos_score, dim=0)

        neg_score  = torch.matmul(input_embedding, negative_embeddings.t()).squeeze()
        neg_score = F.logsigmoid(-neg_score)
        neg_score = torch.sum(neg_score, dim=0)

        return -(pos_score + neg_score)


if __name__ == '__main__':

    ##### PARAMS ####
    embedding_dim = 10
    vocab_size = 322
    context_window = 2
    negative_words = 10
    #################
    positive_matrix = []
    target_context_pair = []


    sentences = SentenceIterator("data/wa/dev.en")
    vocab = Vocabulary(sentences, max_size = vocab_size)

    vocabulary = []
    occurance = Counter()

    for each_sentence in sentences:
        for each_word in each_sentence:
            if each_word not in vocabulary:
                vocabulary.append(each_word)
            occurance[each_word] += 1 # calculates frequency of each word

    i2w = {i: w for i, w in enumerate(vocabulary)}
    w2i = {w: i for i, w in enumerate(vocabulary)}

    def one_hot_negative(index_list):
        one_hot_list = tuple(vocab.one_hot(i2w[i]) for i in index_list)
        # print(one_hot_list)
        neg = torch.stack(one_hot_list, dim = 1)
        return neg

    ea = SkipGramModel(vocab, embedding_dim)

    optmizer = optim.Adam([ea.input_embeddings, ea.output_embeddings])

    for each_sentence in sentences:
        for center_idx, center_word in enumerate(each_sentence):
            if center_word not in vocab.index:
                continue
            center_word_vector = vocab.one_hot(center_word)

            # get pos and neg context words
            context_window_list = get_positive_context(each_sentence, center_idx, context_window)
            for word in context_window_list:
                positive_matrix.append(vocab.one_hot(word))
                negative_samples = list(np.random.randint(0, vocab.N , negative_words))
                target_context_pair.append((center_idx, w2i[word], negative_samples))

    # pprint(target_context_pair)

    # inp = torch.zeros(vocab.N)
    # inp[vocab["the"]] = 1
    # pos = torch.zeros(3, vocab.N)
    # neg = torch.zeros(4, vocab.N)
    for c, p, n in target_context_pair:
        center = vocab.one_hot(i2w[c])
        pos = vocab.one_hot(i2w[p])
        neg = one_hot_negative(n)
        x = ea.forward(center, pos, neg)
        print(x)
