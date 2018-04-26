## The Embed-Align model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distb
from torch.autograd import Variable
from data import SentenceIterator, Vocabulary, get_context_words

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
        negative_embeddings = torch.matmul(negative_words, self.output_embeddings)

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
    context_window = 5
    negative_words = 10
    #################


    sentences = SentenceIterator("data/wa/dev.en")
    vocab = Vocabulary(sentences, max_size = vocab_size)

    ea = SkipGramModel(vocab, embedding_dim)

    optmizer = optim.Adam([ea.input_embeddings, ea.output_embeddings])

    for sentence in sentences:
        for center_idx, center_word in enumerate(sentence):
            if center_vec not in vocab.index:
                continue
            center_vec = vocab.one_hot(center_word)
            positive_matrix = []
            for word in get_context_words(sentence, center_idx, context_window):
                positive_matrix.append(vocab.one_hot(word))

            # TODO sample negative words
            





    inp = torch.zeros(vocab.N)
    inp[vocab_en["the"]] = 1
    pos = torch.zeros(3, vocab.N)
    neg = torch.zeros(4, vocab.N)
    ea.forward(inp, pos, neg)
