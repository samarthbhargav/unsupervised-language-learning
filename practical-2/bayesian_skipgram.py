
# encoder -> word in context
# L, S -> word without context (prior)

## KL - 3.4 on the paper
# use the output of the left (from the dist ) i.e L and S - this is where the other
# product of KL of  gaussians elementwise
# KL should be a positive scalar (check with an assert here)

## The Embed-Align model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distb
from torch.autograd import Variable

import numpy as np

from data import SentenceIterator, Vocabulary, get_context_words
import utils

class BayesianSkipGramModel(nn.Module):
    def __init__(self, vocab, embedding_dim, z_embedding_dim, cuda=False, random_state=None):
        super(BayesianSkipGramModel, self).__init__()

        if random_state:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.vocab = vocab
        self.cuda = cuda

        self.embedding_dim = embedding_dim
        self.z_embedding_dim = z_embedding_dim

        # glorot-initialization
        inference_embedding = nn.init.xavier_uniform_(torch.zeros(self.embedding_dim, self.vocab.N))
        self.inference_embedding = nn.Parameter(inference_embedding, requires_grad = True)

        self.inference_mean_transform = nn.Linear(2*self.embedding_dim, self.z_embedding_dim, bias = True)
        self.inference_sigma_transform = nn.Linear(2*self.embedding_dim, self.z_embedding_dim, bias = True)
        self.inference_transform = nn.Linear(self.z_embedding_dim, vocab.N, bias = True)

        prior_mean = nn.init.xavier_uniform_(torch.zeros(self.z_embedding_dim, self.vocab.N))
        self.prior_mean = nn.Parameter(prior_mean, requires_grad = True)

        prior_sigma = nn.init.xavier_uniform_(torch.zeros(self.z_embedding_dim, self.vocab.N))
        self.prior_sigma = nn.Parameter(prior_sigma)

        self.standard_normal = distb.MultivariateNormal(torch.zeros(self.z_embedding_dim), torch.diag(torch.ones(self.z_embedding_dim)))


        if cuda:
            self.inference_embedding = self.inference_embedding.cuda()
            self.inference_mean_transform = self.inference_mean_transform.cuda()
            self.inference_sigma_transform = self.inference_sigma_transform.cuda()
            self.inference_transform = self.inference_transform.cuda()

            self.prior_mean = self.prior_mean.cuda()
            self.prior_sigma = self.prior_sigma.cuda()


    def forward(self, center_word, context_words):
        ## The inference model / encoder:
        # need one hot encoded vectors of x, and all context vectors
        # E = d x V , d = dim of embedding, V = vocabulary (column vectors everywhere)
        # one_hot(x) times E -> to get dx1 vector (embedding) - deterministically
        # concat every pair of (center_word, context word) - which is 1 x 2d
        # apply elementwise relu on this
        # sum every pair into a single (2d, 1) vector
        # d_z <- embedding size of the latent variable z
        # from the (2d, 1) vector, predict the mean and sigma using 2 (different) feed forward affine layer (paper mentions log sigma squared, but use sigma here)

        if cuda:
            center_word = center_word.cuda()
            context_words = context_words.cuda()

        center_word_embedding = torch.matmul(center_word, self.inference_embedding.t())
        context_words = torch.matmul(context_words, self.inference_embedding.t())
        concated_embeddings = torch.cat((center_word_embedding.repeat(context_words.size(0), 1), context_words), 1)

        summed_embeddings = F.relu(concated_embeddings).sum(0)

        z_mean = self.inference_mean_transform(summed_embeddings)
        z_sigma = F.softplus(self.inference_sigma_transform(summed_embeddings))

        z = z_mean + torch.mul(self.standard_normal.sample(), z_sigma)

        # TODO replace with a different softmax
        f_theta = F.softmax(self.inference_transform(z), dim=0)

        ## 'decoder' -> z to categorial (dim = |V|)
        # each word in the vocab, L (location - means of each word) : a (d, V) matrix
        # multiply L with one_hot(x) -> mu_x
        # each word in the vocab, S(d, V) matrix
        # multiply S with one_hot(x) -> s_z, softplus on top of that to finally get sigma
        # square it and put in on the diagonal to get the multivariate sigma
        # sample Z from the \mu_x and \sigma_x multivariate
        # take z and project using NN, softmax on last layer to get a categorical dist over the vocabulary
        # loss is negative of the ELBO
        # MC estimate the first term which is the expectation p(c | z, x) of one sample
        # sample one variational sample (from the encoder)
        # z = \mu_{encoder} + noise (*) \sigma_{encoder}
        # use this z to the decoder
        # sum for every context word, compute the log probability

        prior_mean = torch.matmul(self.prior_mean, center_word)
        prior_sigma = F.softplus(torch.matmul(self.prior_sigma, center_word))

        return f_theta, z_mean, z_sigma, prior_mean, prior_sigma

def compute_loss(f_theta, z_mean, z_sigma, prior_mean, prior_sigma, context_idx):
    prior = distb.MultivariateNormal(prior_mean, torch.diag(prior_sigma))
    posterior = distb.MultivariateNormal(z_mean, torch.diag(z_sigma))

    log_pos = torch.empty(1)
    for idx in context_idx:
        log_pos += torch.log(f_theta[idx])

    kl_term = distb.kl_divergence(posterior, prior).sum()
    loss = kl_term - log_pos
    return loss

if __name__ == '__main__':

    ##### PARAMS ####
    embedding_dim = 10
    z_embedding_dim = 15
    vocab_size = 1000
    context_window = 5
    negative_words = 10
    n_epochs = 1
    random_state = 42
    data_path = "data/europarl/training.en"
    use_cuda = True
    #################
    sentences = SentenceIterator(data_path)
    vocab = Vocabulary(sentences, max_size = vocab_size)

    bsm = BayesianSkipGramModel(vocab, embedding_dim, z_embedding_dim, random_state=random_state, cuda=use_cuda)

    optimizer = optim.Adam(bsm.parameters())

    loss = []
    tictoc = utils.TicToc()
    for epoch in range(1, n_epochs + 1):
        print("Running epoch: ", epoch)
        epoch_loss = []
        for sentence_num, sentence in enumerate(sentences):
            if sentence_num % 100 == 0:
                print("Sentence: ", sentence_num)
                tictoc.tic()
            for center_idx, center_word in enumerate(sentence):
                if center_word not in vocab.index:
                    continue
                center_vec = vocab.one_hot(center_word)
                context_words = []
                context_idx = []
                for word in get_context_words(sentence, center_idx, context_window):
                    if word not in vocab.index:
                        continue
                    context_idx.append(vocab[word])
                    context_words.append(vocab.one_hot(word))
                if len(context_words) == 0:
                    continue
                optimizer.zero_grad()
                f_theta, z_mean, z_sigma , prior_mean, prior_sigma = bsm(torch.FloatTensor(center_vec), torch.FloatTensor(context_words))
                val, idx = f_theta.max(0)
                val = idx.data.numpy()
                if epoch % 5 == 0:
                    print(center_word, vocab.word(int(val)))
                loss = compute_loss(f_theta, z_mean, z_sigma, prior_mean, prior_sigma, context_idx)
                epoch_loss.append(loss.data.numpy()[0])
                loss.backward()
                optimizer.step()

        print(np.mean(epoch_loss))
