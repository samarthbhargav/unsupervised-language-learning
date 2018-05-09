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

        model = SkipGramModel(vocab, params["embedding_dim"], params["use_cuda"])
        model.load_state_dict(torch.load(os.path.join(path, model_name)))

        return (model, loss, params)

    def __init__(self, vocab, embedding_dim, use_cuda=True):
        super(SkipGramModel, self).__init__()

        self.vocab = vocab

        self.embedding_dim = embedding_dim

        self.input_embeddings = nn.Embedding(vocab.N, self.embedding_dim, sparse=True)
        self.output_embeddings = nn.Embedding(vocab.N, self.embedding_dim, sparse=True)

        self.use_cuda = use_cuda

        if use_cuda:
            self.input_embeddings = self.input_embeddings.cuda()
            self.output_embeddings = self.output_embeddings.cuda()


    def forward(self, center_words, positive_words, negative_words):

        if self.use_cuda:
            center_words = center_words.cuda()
            positive_words = positive_words.cuda()
            negative_words = negative_words.cuda()

        center_embedding = self.input_embeddings(center_words)
        positive_embeddings = self.output_embeddings(positive_words)
        negative_embeddings = self.output_embeddings(negative_words)

        batch_size = center_words.size(0)

        #pos = center_embedding @ positive_embeddings.t()
        # from https://discuss.pytorch.org/t/dot-product-batch-wise/9746/3
        center_pos_dot = torch.bmm(center_embedding.view(batch_size, 1, self.embedding_dim), positive_embeddings.view(batch_size, embedding_dim, 1))
        log_pos = - F.logsigmoid(center_pos_dot)

        # each row = 1 word's negative embeddings
        center_neg_dot = torch.matmul(center_embedding, negative_embeddings.t())
        log_neg = - F.logsigmoid(center_neg_dot)
        log_neg = log_neg.sum(1)

        return (log_pos + log_neg).mean()


    def get_embeddings(self):
        ematrix = np.zeros((self.vocab.N, self.embedding_dim))
        words = []
        for eidx, (word, idx) in enumerate(self.vocab.index.items()):
            inp = torch.LongTensor(np.array([idx]))
            if self.use_cuda:
                inp = inp.cuda()
            temp = self.input_embeddings(inp)
            temp2 = temp.cpu()
            temp3 = temp2.detach().numpy()
            ematrix[eidx] = temp3
            words.append(word)
        return ematrix, words

    def save_model(self, model_root, model_name, loss, params):
        path = os.path.join(model_root, model_name)
        utils.mkdir(path)

        torch.save(self.state_dict(), os.path.join(path, model_name))
        utils.save_object(self.vocab, os.path.join(path, model_name + "_vocab"))
        utils.save_object(loss, os.path.join(path, model_name + "_loss"))
        utils.save_object(params, os.path.join(path, model_name + "_params"))

def get_negative_batch(vocab, n_negative, positive_words):
    neg = []
    while len(neg) < n_negative:
        neg_word = vocab.ust.sample()
        if neg_word not in vocab.index or neg_word in positive_words:
            continue
        neg.append(vocab[neg_word])
    return np.array(neg)


def batch_iterator(sentences, vocab, batch_size, n_negative):
    batch_center = np.zeros(batch_size)
    batch_context = np.zeros(batch_size)
    batch_index = 0
    batch_pos_words = set()

    batch_number = 0
    for sentence_num, each_sentence in enumerate(sentences):
        for center_idx, center_word in enumerate(each_sentence):
            if center_word not in vocab.index:
                continue

            context_window_list = get_context_words(vocab.process(each_sentence), center_idx, context_window)

            for positive_word in context_window_list:
                # sub-sampling
                if vocab.ust.remove_word(positive_word):
                    continue

                batch_center[batch_index] = vocab[center_word]
                batch_context[batch_index] = vocab[positive_word]
                batch_pos_words.add(positive_word)
                batch_index += 1
                if batch_index == batch_size:
                    negative_words = get_negative_batch(vocab, n_negative, batch_pos_words)
                    yield torch.LongTensor(batch_center), torch.LongTensor(batch_context), torch.LongTensor(negative_words)

                    batch_center = np.zeros(batch_size)
                    batch_context = np.zeros(batch_size)
                    batch_index = 0
                    batch_pos_words = set()

                    batch_number += 1
                    if batch_number % 5000 == 0:
                        tictoc.tic("Sentence: {} of {}. Batches Processed: {}".format(sentence_num + 1, vocab.sentence_count, batch_number))

    # last batch
    if batch_index > 0:
        yield torch.LongTensor(batch_center), torch.LongTensor(batch_context), torch.LongTensor(negative_words)
        tictoc.tic("Sentence: {} of {}. Batches Processed: {}".format(sentence_num + 1, vocab.sentence_count, batch_number))


if __name__ == '__main__':

    ##### PARAMS ####
    params = {
        "embedding_dim" : 200,
        "vocab_size": 10000,
        "context_window": 5,
        "n_negative": 5,
        "model_name": "test",
        "stop_words_file": None, # use sub-sampling instead
        "n_epochs": 10,
        "data_path": "data/hansards/training.en",
        "use_cuda": False,
        "batch_size": 500,
    }
    #################
    locals().update(params)

    stop_words = None
    if stop_words_file:
        stop_words = read_stop_words(stop_words_file)

    sentences = SentenceIterator(data_path, stop_words=stop_words)
    vocab = Vocabulary(sentences, max_size = vocab_size)

    sgm = SkipGramModel(vocab, embedding_dim, use_cuda=use_cuda)
    optimizer = optim.SparseAdam(sgm.parameters())

    tictoc = utils.TicToc()
    epoch_losses = []

    for epoch in np.arange(1, n_epochs + 1):
        print("Running epoch: ", epoch)
        epoch_loss = utils.Mean()
        for batch in batch_iterator(sentences, vocab, batch_size, n_negative):
            batch_center, batch_context, negative_words = batch
            optimizer.zero_grad()
            loss = sgm.forward(batch_center, batch_context, negative_words)
            epoch_loss.add(loss.item())
            loss.backward()
            optimizer.step()

        epoch_losses.append(epoch_loss.mean())
        tictoc.tic("Epoch complete: Mean loss: {}".format(epoch_loss.mean()))

    sgm.save_model("./models", model_name, epoch_losses, params)
