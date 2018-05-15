# encoder -> word in context
# L, S -> word without context (prior)

## KL - 3.4 on the paper
# use the output of the left (from the dist ) i.e L and S - this is where the other
# product of KL of  gaussians elementwise
# KL should be a positive scalar (check with an assert here)

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distb
from torch.autograd import Variable

import numpy as np

from data import SentenceIterator, Vocabulary, get_context_words, read_stop_words
import utils



class BayesianSkipgram(nn.Module):

    @staticmethod
    def load(model_root, model_name):
        path = os.path.join(model_root, model_name)

        loss = utils.load_object(os.path.join(path, model_name + "_loss"))
        params = utils.load_object(os.path.join(path, model_name + "_params"))
        vocab = utils.load_object(os.path.join(path, model_name + "_vocab"))

        model = BayesianSkipgram(vocab, params["z_embedding_dim"])
        model.load_state_dict(torch.load(os.path.join(path, model_name)))

        return (model, loss, params)


    def __init__(self, vocab, embedding_dim, use_cuda=False):
        super(BayesianSkipgram, self).__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.use_cuda = use_cuda

        # Inference model
        self.inference_embedding = nn.Embedding(vocab.N, embedding_dim)
        self.inference_affine = nn.Linear(embedding_dim*2, embedding_dim)
        self.inference_mean = nn.Linear(embedding_dim, embedding_dim)
        self.inference_sigma = nn.Linear(embedding_dim, embedding_dim)

        # Generative model
        self.generative_mean = nn.Embedding(vocab.N, embedding_dim)
        self.generative_sigma = nn.Embedding(vocab.N, embedding_dim)
        self.generative_affine = nn.Linear(embedding_dim, vocab.N)

        std_mean = torch.zeros(self.embedding_dim)
        std_cov = torch.diag(torch.ones(self.embedding_dim))

        if use_cuda:
            std_mean = std_mean.cuda()
            std_cov = std_cov.cuda()
        self.standard_normal = distb.MultivariateNormal(std_mean, std_cov)


        if use_cuda:
            self.inference_embedding = self.inference_embedding.cuda()
            self.inference_affine = self.inference_affine.cuda()
            self.inference_mean = self.inference_mean.cuda()
            self.inference_sigma = self.inference_sigma.cuda()
            self.generative_mean = self.generative_mean.cuda()
            self.generative_sigma = self.generative_sigma.cuda()
            self.generative_affine = self.generative_affine.cuda()

    def get_distribution(self, x, context_words):
        context_words = context_words.view(1, -1)
        context_size = context_words.size(1)

        # Inference model
        x_stacked = x.repeat(context_size, 1).transpose(0, 1)
        center_embedding = self.inference_embedding(x_stacked)
        context_embedding = self.inference_embedding(context_words)

        emb = F.relu(self.inference_affine(torch.cat([center_embedding, context_embedding], -1)))
        emb_sum = emb.sum(1)

        inf_mu = self.inference_mean(emb_sum)
        inf_sigma = F.softplus(self.inference_sigma(emb_sum))

        # Sample
        z = inf_mu + torch.mul(self.standard_normal.sample(), inf_sigma)

        mu = self.generative_mean(x)
        sigma = F.softplus(self.generative_sigma(x))

        return mu, sigma, inf_mu, inf_sigma, z


    def forward(self, x_batch, context_words_batch):

        if self.use_cuda:
            x_batch = x_batch.cuda()
            context_words_batch = context_words_batch.cuda()

        total_loss = torch.zeros(x_batch.size(0))
        context_size = context_words_batch.size(1)

        if self.use_cuda:
            total_loss = total_loss.cuda()

        for idx, (x, context_words) in enumerate(zip(x_batch, context_words_batch)):
            mu, sigma, inf_mu, inf_sigma, z = self.get_distribution(x, context_words)
            # Generative model
            logprobs = F.log_softmax(self.generative_affine(z), dim=-1).squeeze(0)

            # Loss
            loss_probs = torch.zeros_like(context_words).type(torch.FloatTensor)
            if self.use_cuda:
                loss_probs = loss_probs.cuda()

            for i, context_word in enumerate(context_words):
                loss_probs[i] = logprobs[context_word]

            reconstruction_loss = loss_probs.sum(-1)
            kl_loss = torch.log(sigma/inf_sigma) + (inf_sigma.pow(2) + (inf_mu - sigma).pow(2)) / (2*sigma.pow(2)) - 0.5
            kl_loss = kl_loss.sum()

            loss = kl_loss - reconstruction_loss
            if np.isnan(loss.item()):
                print(kl_loss, reconstruction_loss, loss)
            total_loss[idx] = loss

        return total_loss.mean()

    def save_model(self, model_root, model_name, loss, params):
        path = os.path.join(model_root, model_name)
        utils.mkdir(path)

        torch.save(self.state_dict(), os.path.join(path, model_name))
        utils.save_object(self.vocab, os.path.join(path, model_name + "_vocab"))
        utils.save_object(loss, os.path.join(path, model_name + "_loss"))
        utils.save_object(params, os.path.join(path, model_name + "_params"))


def batch_iterator(sentences, vocab, batch_size, context_size):
    batch_center = np.zeros(batch_size)
    batch_context = np.zeros((batch_size, context_size))
    batch_index = 0
    batch_number = 0
    for sentence_num, sentence in enumerate(sentences):
        for center_idx, center_word in enumerate(sentence):
            if center_word not in vocab.index:
                continue

            batch_center[batch_index] = vocab[center_word]
            # needs to be the same size, so pad it!
            context_window_list = get_context_words(vocab.process(sentence), center_idx, context_window, pad=True)
            for idx, positive_word in enumerate(context_window_list):
                batch_context[batch_index, idx] = vocab[positive_word]
            batch_index += 1

            if batch_index == batch_size:
                yield torch.LongTensor(batch_center), torch.LongTensor(batch_context)

                batch_center = np.zeros(batch_size)
                batch_context = np.zeros((batch_size, context_size))
                batch_index = 0

                batch_number += 1
                if batch_number % 10 == 0:
                    tictoc.tic("Sentence: {} of {}. Batches Processed: {}".format(sentence_num + 1, vocab.sentence_count, batch_number))

    # last batch
    if batch_index > 0:
        yield torch.LongTensor(batch_center), torch.LongTensor(batch_context)
        tictoc.tic("Sentence: {} of {}. Batches Processed: {}".format(sentence_num + 1, vocab.sentence_count, batch_number))


if __name__ == '__main__':

    ##### PARAMS ####
    params = {
        "embedding_dim": 200,
        "z_embedding_dim" : 2000,
        "vocab_size" : 10000,
        "context_window" : 7,
        "n_epochs" : 3,
        "random_state" : 42,
        "data_path" : "data/wa/test.en",
        "stop_words_file" : "data/en_stopwords.txt",
        "model_name" : "test_bn",
        "use_cuda" : False,
        "batch_size": 32
    }
    #################
    locals().update(params)

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    stop_words = None
    if stop_words_file:
        stop_words = read_stop_words(stop_words_file)


    sentences = SentenceIterator(data_path, stop_words=stop_words)
    vocab = Vocabulary(sentences, max_size = vocab_size)

    #bsm = BayesianSkipGramModel(vocab, embedding_dim, z_embedding_dim, random_state=random_state, cuda=use_cuda)
    bsm = BayesianSkipgram(vocab, z_embedding_dim, use_cuda)

    # set up model parameters to learn
    optimizer = optim.Adam(bsm.parameters(), lr=1e-4)

    epoch_losses = []
    tictoc = utils.TicToc()
    for epoch in range(1, n_epochs + 1):
        print("Running epoch: ", epoch)
        epoch_loss = utils.Mean()

        for batch_number, batch in enumerate(batch_iterator(sentences, vocab, batch_size, context_window)):

            if batch_number % 10 == 0:
                tictoc.tic("Batches Processed: {} , Mean Loss: {}".format(batch_number + 1, epoch_loss.mean()))


            optimizer.zero_grad()

            loss = bsm(batch[0], batch[1])

            if use_cuda:
                epoch_loss.add(loss.cpu().item())
            else:
                epoch_loss.add(loss.item())

            loss.backward()
            optimizer.step()

        epoch_losses.append(epoch_loss.mean())
        tictoc.tic("Epoch complete: Mean loss: {}".format(epoch_loss.mean()))

    bsm.save_model("./models", model_name, epoch_losses, params)

    del bsm

    BayesianSkipgram.load("./models", model_name)
    # model_save_path = os.path.join("./models/", model_name)
    # print("Saving model: ", model_save_path)
    # torch.save(bsm.state_dict(), model_save_path)
    # utils.save_object(epoch_losses, os.path.join("./models/", model_name) + "_loss")
