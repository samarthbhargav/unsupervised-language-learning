## The Skip-Align model

import os
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distb
from torch.autograd import Variable

import numpy as np

from data import SentenceIterator, Vocabulary, get_context_words


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


        # print(input_embedding.size())
        # print("Input embedding:", input_embedding.size(), "Positive :", positive_embeddings.size())
        pos_score  = torch.matmul(input_embedding, positive_embeddings.t()).squeeze()
        pos_score = F.logsigmoid(pos_score)
        pos_score = torch.sum(pos_score, dim=0)

        negative_embeddings = torch.matmul(negative_words, self.output_embeddings)

        neg_score  = torch.matmul(input_embedding, negative_embeddings.t()).squeeze()
        neg_score = F.logsigmoid(-neg_score)
        neg_score = torch.sum(neg_score, dim=0)

        return -(pos_score + neg_score)

def one_hot_negative(index_list, vocab):
    neg = np.zeros((len(index_list), vocab.N))
    for row, idx in enumerate(index_list):
        neg[row] = vocab.one_hot(vocab.word(idx))
    return neg


if __name__ == '__main__':

    ##### PARAMS ####
    embedding_dim = 10
    vocab_size = 322
    context_window = 2
    negative_words = 15
    model_name = "test"
    #################
    positive_matrix = []
    target_context_pair = []

    sentences = SentenceIterator("data/wa/dev.en")
    vocab = Vocabulary(sentences, max_size = vocab_size)

    ea = SkipGramModel(vocab, embedding_dim)

    optimizer = optim.Adam([ea.input_embeddings, ea.output_embeddings])

    for each_sentence in sentences:
        for center_idx, center_word in enumerate(each_sentence):
            if center_word not in vocab.index:
                continue
            center_word_vector = vocab.one_hot(center_word)

            context_window_list = get_context_words(each_sentence, center_idx, context_window)
            for word in context_window_list:
                positive_matrix.append(vocab.one_hot(word))

            # TODO change this to sample from words which don't co-occur with target word
            negative_samples = np.random.randint(0, vocab.N , negative_words)
            negative_matrix = one_hot_negative(negative_samples, vocab)

            optimizer.zero_grad()
            loss = ea.forward(torch.FloatTensor(center_word_vector), torch.FloatTensor(positive_matrix), torch.FloatTensor(negative_matrix))
            loss.backward()
            optimizer.step()

    model_save_path = os.path.join("./models/", model_name)
    print("Saving model: ", model_save_path)
    torch.save(ea.state_dict(), model_save_path)
