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

from data import SentenceIterator, Vocabulary, get_context_words, read_stop_words
import utils

class SkipGramModel(nn.Module):

    @staticmethod
    def load(model_root, model_name):
        path = os.path.join(model_root, model_name)

        loss = utils.load_object(os.path.join(path, model_name + "_loss"))
        params = utils.load_object(os.path.join(path, model_name + "_params"))
        vocab = utils.load_object(os.path.join(path, model_name + "_vocab"))

        model = SkipGramModel(vocab, params["embedding_dim"])
        model.load_state_dict(torch.load(os.path.join(path, model_name)))

        return (model, loss, params)

    def __init__(self, vocab, embedding_dim):
        super(SkipGramModel, self).__init__()

        self.vocab = vocab

        self.embedding_dim = embedding_dim

        # # TODO add noise
        # self.input_embeddings = torch.zeros(self.embedding_dim, vocab.N)
        # self.input_embeddings += torch.rand(self.input_embeddings.size())
        # self.input_embeddings = nn.Parameter(self.input_embeddings)
        #
        # self.output_embeddings = torch.zeros(vocab.N, self.embedding_dim)
        # self.output_embeddings += torch.rand(self.output_embeddings.size())
        #
        # self.output_embeddings = nn.Parameter(self.output_embeddings)
        self.input_embeddings = nn.Embedding(vocab.N, self.embedding_dim, sparse=True)
        self.output_embeddings = nn.Embedding(vocab.N, self.embedding_dim, sparse=True)


    def forward(self, center_word, positive_words, negative_words):
        # input_embedding = torch.matmul(center_word, self.input_embeddings.t()).view(1, -1)
        # positive_embeddings = torch.matmul(positive_words, self.input_embeddings.t())
        # negative_embeddings = torch.matmul(negative_words, self.output_embeddings)
        input_embedding = self.input_embeddings(center_word)
        positive_embeddings = self.output_embeddings(positive_words)

        score  = torch.mul(input_embedding, positive_embeddings)
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score).squeeze()

        negative_embeddings = self.output_embeddings(negative_words)

        neg_score = torch.matmul(input_embedding, negative_embeddings.t()).squeeze()

        neg_score = torch.sum(neg_score)

        neg_score = F.logsigmoid(-1*neg_score).squeeze()

        loss = score + neg_score

        return -1*loss.sum()

    def get_embeddings(self):
        ematrix = np.zeros((self.vocab.N, self.embedding_dim))
        words = []
        for eidx, (word, idx) in enumerate(self.vocab.index.items()):
            ematrix[eidx] = self.input_embeddings(torch.LongTensor(np.array([idx]))).detach().numpy()
            words.append(word)
        return ematrix, words

    def save_model(self, model_root, model_name, loss, params):
        path = os.path.join(model_root, model_name)
        utils.mkdir(path)

        torch.save(self.state_dict(), os.path.join(path, model_name))
        utils.save_object(self.vocab, os.path.join(path, model_name + "_vocab"))
        utils.save_object(loss, os.path.join(path, model_name + "_loss"))
        utils.save_object(params, os.path.join(path, model_name + "_params"))

def get_negative_matrix(vocab, n_negative):
    neg = []
    while len(neg) < n_negative:
        neg_word = vocab.ust.sample()
        if neg_word not in vocab.index:
            continue
        neg.append(vocab[neg_word])
    return np.array(neg)

if __name__ == '__main__':

    ##### PARAMS ####
    params = {
        "embedding_dim" : 10,
        "vocab_size": 10000,
        "context_window": 2,
        "n_negative": 5,
        "model_name": "test",
        "stop_words_file": None, # use sub-sampling instead
        "n_epochs": 5,
        "data_path": "data/hansards/training.en"
    }
    #################


    # this is not safe
    locals().update(params)

    stop_words = None
    if stop_words_file:
        stop_words = read_stop_words(stop_words_file)

    sentences = SentenceIterator(data_path, stop_words=stop_words)
    vocab = Vocabulary(sentences, max_size = vocab_size)

    sgm = SkipGramModel(vocab, embedding_dim)
    sgm.get_embeddings()
    optimizer = optim.SparseAdam(sgm.parameters())

    tictoc = utils.TicToc()
    epoch_losses = []

    for epoch in np.arange(1, n_epochs + 1):
        print("Running epoch: ", epoch)
        epoch_loss = []
        for sentence_num, each_sentence in enumerate(sentences):
            if sentence_num % 100 == 0:
                tictoc.tic("Sentence: {} of {}".format(sentence_num + 1, vocab.sentence_count))

            for center_idx, center_word in enumerate(each_sentence):
                if center_word not in vocab.index:
                    continue

                center_word_vector = vocab.one_hot(center_word)

                context_window_list = get_context_words(vocab.process(each_sentence), center_idx, context_window)

                positive_matrix = []
                for word in context_window_list:
                    # sub-sampling
                    if vocab.ust.remove_word(word):
                        continue
                    positive_matrix.append(vocab[word])

                if len(positive_matrix) == 0:
                    continue


                negative_matrix = get_negative_matrix(vocab, n_negative)

                optimizer.zero_grad()
                loss = sgm.forward(torch.LongTensor(np.array([vocab[center_word]], dtype=np.long)),
                                    torch.LongTensor(np.array(positive_matrix)), torch.LongTensor(negative_matrix))
                epoch_loss.append(loss.cpu().data.numpy())
                loss.backward()
                optimizer.step()

        epoch_losses.append(np.mean(epoch_loss))
        tictoc.tic("Epoch complete: Mean loss: {}".format(np.mean(epoch_loss)))

    sgm.save_model("./models", model_name, epoch_losses, params)
