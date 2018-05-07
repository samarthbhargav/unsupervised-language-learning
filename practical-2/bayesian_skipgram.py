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

    def forward(self, x, context_words):
        if self.use_cuda:
            x = x.cuda()
            context_words = context_words.cuda()

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

        # Generative model
        logprobs = F.log_softmax(self.generative_affine(z), dim=-1).squeeze(0)
        #print(logprobs.size())
        mu = self.generative_mean(x)
        sigma = F.softplus(self.generative_sigma(x))

        # Loss
        loss_probs = torch.zeros_like(context_words).type(torch.FloatTensor)
        if self.use_cuda:
            loss_probs = loss_probs.cuda()

        # for i in range(batch_size):
        #     for j in range(context_size):
        #         loss_probs[i, j] = logprobs[i, context_words[i, j]]
        for i, context_word in enumerate(context_words):
            loss_probs[i] = logprobs[context_word]

        reconstruction_loss = loss_probs.sum(-1)
        kl_loss = torch.log(sigma/inf_sigma) + (inf_sigma.pow(2) + (inf_mu - sigma).pow(2)) / (2*sigma.pow(2)) - 0.5
        kl_loss = kl_loss.sum()

        #print(reconstruction_loss, kl_loss)

        loss = kl_loss - reconstruction_loss
        return loss.mean()

    def save_model(self, model_root, model_name, loss, params):
        path = os.path.join(model_root, model_name)
        utils.mkdir(path)

        torch.save(self.state_dict(), os.path.join(path, model_name))
        utils.save_object(self.vocab, os.path.join(path, model_name + "_vocab"))
        utils.save_object(loss, os.path.join(path, model_name + "_loss"))
        utils.save_object(params, os.path.join(path, model_name + "_params"))

class BayesianSkipGramModel(nn.Module):

    @staticmethod
    def load(model_root, model_name):
        path = os.path.join(model_root, model_name)

        loss = utils.load_object(os.path.join(path, model_name + "_loss"))
        params = utils.load_object(os.path.join(path, model_name + "_params"))
        vocab = utils.load_object(os.path.join(path, model_name + "_vocab"))

        model = BayesianSkipGramModel(vocab, params["embedding_dim"], params["z_embedding_dim"], params["use_cuda"])
        model.load_state_dict(torch.load(os.path.join(path, model_name)))

        return (model, loss, params)

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
        if cuda:
            inference_embedding = inference_embedding.cuda()
        self.inference_embedding = nn.Parameter(inference_embedding, requires_grad = True)

        self.inference_mean_transform = nn.Linear(2*self.embedding_dim, self.z_embedding_dim, bias = True)
        self.inference_sigma_transform = nn.Linear(2*self.embedding_dim, self.z_embedding_dim, bias = True)


        prior_mean = nn.init.xavier_uniform_(torch.zeros(self.z_embedding_dim, self.vocab.N))
        if cuda:
            prior_mean = prior_mean.cuda()
        self.prior_mean = nn.Parameter(prior_mean, requires_grad = True)

        prior_sigma = nn.init.xavier_uniform_(torch.zeros(self.z_embedding_dim, self.vocab.N))
        if cuda:
            prior_sigma = prior_sigma.cuda()
        self.prior_sigma = nn.Parameter(prior_sigma)

        self.generative_transform = nn.Linear(self.z_embedding_dim, vocab.N, bias = True)

        std_mean = torch.zeros(self.z_embedding_dim)
        std_cov = torch.diag(torch.ones(self.z_embedding_dim))

        if cuda:
            std_mean = std_mean.cuda()
            std_cov = std_cov.cuda()
        self.standard_normal = distb.MultivariateNormal(std_mean, std_cov)


        if cuda:
            self.inference_mean_transform = self.inference_mean_transform.cuda()
            self.inference_sigma_transform = self.inference_sigma_transform.cuda()
            self.generative_transform = self.generative_transform.cuda()


    def forward(self, center_word, context_words, context_word_idxs):
        ## The inference model / encoder:
        # need one hot encoded vectors of x, and all context vectors
        # E = d x V , d = dim of embedding, V = vocabulary (column vectors everywhere)
        # one_hot(x) times E -> to get dx1 vector (embedding) - deterministically
        # concat every pair of (center_word, context word) - which is 1 x 2d
        # apply elementwise relu on this
        # sum every pair into a single (2d, 1) vector
        # d_z <- embedding size of the latent variable z
        # from the (2d, 1) vector, predict the mean and sigma using 2 (different) feed forward affine layer (paper mentions log sigma squared, but use sigma here)

        if self.cuda:
            center_word = center_word.cuda()
            context_words = context_words.cuda()

        center_word_embedding = torch.matmul(center_word, self.inference_embedding.t())
        context_words = torch.matmul(context_words, self.inference_embedding.t())
        concated_embeddings = torch.cat((center_word_embedding.repeat(context_words.size(0), 1), context_words), 1)

        summed_embeddings = F.relu(concated_embeddings).sum(0)

        u = self.inference_mean_transform(summed_embeddings)
        s = F.softplus(self.inference_sigma_transform(summed_embeddings))

        z = u + torch.mul(self.standard_normal.sample(), s)
        logprobs = F.log_softmax(self.generative_transform(z), dim=-1)

        z_mean = torch.matmul(center_word, self.prior_mean.t())
        z_sigma = F.softplus(torch.matmul(center_word, self.prior_sigma.t()))

        # Loss
        loss_probs = torch.zeros(len(context_words)).type(torch.FloatTensor)
        if self.cuda:
            loss_probs = loss_probs.cuda()

        for idx, cidx in enumerate(context_word_idxs):
            loss_probs[idx] = logprobs[cidx]

        kl_loss = torch.log(z_sigma/s) + (s.pow(2) + (u - z_sigma).pow(2)) / (2*z_sigma.pow(2)) - 0.5

        loss = loss_probs.sum() - kl_loss.sum()
        return loss


    def save_model(self, model_root, model_name, loss, params):
        path = os.path.join(model_root, model_name)
        utils.mkdir(path)

        torch.save(self.state_dict(), os.path.join(path, model_name))
        utils.save_object(self.vocab, os.path.join(path, model_name + "_vocab"))
        utils.save_object(loss, os.path.join(path, model_name + "_loss"))
        utils.save_object(params, os.path.join(path, model_name + "_params"))

def compute_loss(f_theta, z_mean, z_sigma, prior_mean, prior_sigma, context_idx, cuda):
    prior = distb.MultivariateNormal(prior_mean, torch.diag(torch.mul(prior_sigma, prior_sigma)))
    posterior = distb.MultivariateNormal(z_mean, torch.diag(torch.mul(z_sigma, z_sigma)))

    log_pos = torch.zeros(1)
    if use_cuda:
        log_pos = log_pos.cuda()

    for idx in context_idx:
        log_pos += torch.log(f_theta[idx])

    print(log_pos, "\n", z_mean, "\n", z_sigma, "\n", prior_mean, "\n", prior_sigma)
    kl_term = distb.kl_divergence(posterior, prior).sum()
    print(kl_term)
    #kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var)
    loss = log_pos - kl_term

    return loss

if __name__ == '__main__':

    ##### PARAMS ####
    params = {
        "embedding_dim": 10,
        "z_embedding_dim" : 100,
        "vocab_size" : 10000,
        "context_window" : 7,
        "negative_words" : 10,
        "n_epochs" : 10,
        "random_state" : 42,
        "data_path" : "data/wa/dev.en",
        "stop_words_file" : "data/en_stopwords.txt",
        "model_name" : "test_bn",
        "use_cuda" : False
    }
    #################
    locals().update(params)

    stop_words = None
    if stop_words_file:
        stop_words = read_stop_words(stop_words_file)


    sentences = SentenceIterator(data_path, stop_words=stop_words)
    vocab = Vocabulary(sentences, max_size = vocab_size)

    #bsm = BayesianSkipGramModel(vocab, embedding_dim, z_embedding_dim, random_state=random_state, cuda=use_cuda)
    bsm = BayesianSkipgram(vocab, z_embedding_dim, use_cuda)

    # set up model parameters to learn
    # model_params = [bsm.inference_embedding, bsm.prior_mean, bsm.prior_sigma]
    # model_params.extend(bsm.inference_mean_transform.parameters())
    # model_params.extend(bsm.inference_sigma_transform.parameters())
    # model_params.extend(bsm.generative_transform.parameters())
    optimizer = optim.Adam(bsm.parameters(), lr=1e-4)

    epoch_losses = []
    tictoc = utils.TicToc()
    for epoch in range(1, n_epochs + 1):
        print("Running epoch: ", epoch)
        epoch_loss = utils.Mean()
        for sentence_num, sentence in enumerate(sentences):
            if sentence_num % 100 == 0:
                tictoc.tic("Sentence: {} of {}".format(sentence_num + 1, vocab.sentence_count))

            for center_idx, center_word in enumerate(sentence):
                if center_word not in vocab.index:
                    continue
                center_vec = vocab.one_hot(center_word)
                context_words = []
                context_idx = []
                for word in get_context_words(vocab.process(sentence), center_idx, context_window):
                    if word not in vocab.index:
                        continue
                    context_idx.append(vocab[word])
                    context_words.append(vocab.one_hot(word))
                if len(context_words) == 0:
                    continue
                optimizer.zero_grad()
                #f_theta, z_mean, z_sigma , prior_mean, prior_sigma = bsm(torch.FloatTensor(center_vec), torch.FloatTensor(context_words))
                loss = bsm(torch.LongTensor(np.array([vocab[center_word]])), torch.LongTensor(context_idx))

                #loss = compute_loss(f_theta, z_mean, z_sigma, prior_mean, prior_sigma, context_idx, use_cuda)
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
