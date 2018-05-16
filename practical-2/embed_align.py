import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distb
from torch.autograd import Variable

import numpy as np

from data import SentenceIterator, Vocabulary, get_context_words, read_stop_words
from skipgram import get_negative_batch
import utils
from utils import kl_div


# UNK

# standard prior is used only in KL term
def get_standard_normal(dimension, cuda):
    std_mean = torch.zeros(dimension)
    std_cov = torch.diag(torch.ones(dimension))
    if cuda:
        std_mean = std_mean.cuda()
        std_cov = std_cov.cuda()
    return distb.MultivariateNormal(std_mean, std_cov)

class EmbedAlignModel(nn.Module):

    @staticmethod
    def load(model_root, model_name):
        path = os.path.join(model_root, model_name)

        loss = utils.load_object(os.path.join(path, model_name + "_loss"))
        params = utils.load_object(os.path.join(path, model_name + "_params"))
        vocab_x = utils.load_object(os.path.join(path, model_name + "_vocab_x"))
        vocab_y = utils.load_object(os.path.join(path, model_name + "_vocab_y"))

        model = EmbedAlignModel(vocab_x, vocab_y, params["embedding_dim"],
                random_state=params["random_state"], use_cuda=params["use_cuda"])
        model.load_state_dict(torch.load(os.path.join(path, model_name)))
        return (model, loss, params)

    def save_model(self, model_root, model_name, loss, params):
        path = os.path.join(model_root, model_name)
        utils.mkdir(path)
        torch.save(self.state_dict(), os.path.join(path, model_name))
        utils.save_object(self.vocab_x, os.path.join(path, model_name + "_vocab_x"))
        utils.save_object(self.vocab_y, os.path.join(path, model_name + "_vocab_y"))
        utils.save_object(loss, os.path.join(path, model_name + "_loss"))
        utils.save_object(params, os.path.join(path, model_name + "_params"))



    def __init__(self, vocab_x, vocab_y, embedding_dim, random_state=None, use_cuda=True):
        super(EmbedAlignModel, self).__init__()
        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.use_cuda = use_cuda

        if random_state:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        self.embedding_dim = embedding_dim

        self.internal_embedding = nn.Embedding(vocab_x.N, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, embedding_dim,
                num_layers=1, bidirectional=True)

        # inference
        self.mu_affine_1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.mu_affine_2 = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.sigma_affine_1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.sigma_affine_2 = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.x_affine_1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.x_affine_2 = nn.Linear(self.embedding_dim, self.vocab_x.N)

        self.y_affine_1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.y_affine_2 = nn.Linear(self.embedding_dim, self.vocab_y.N)


        # deteministic embedding used for the softmax approximation
        self.softmax_approx_x = nn.Embedding(self.vocab_x.N, self.embedding_dim)
        self.softmax_approx_y = nn.Embedding(self.vocab_y.N, self.embedding_dim)

        std_mean = torch.zeros(self.embedding_dim)
        std_cov = torch.diag(torch.ones(self.embedding_dim))
        if use_cuda:
            std_mean = std_mean.cuda()
            std_cov = std_cov.cuda()

            self.internal_embedding = self.internal_embedding.cuda()
            self.lstm = self.lstm.cuda()
            self.mu_affine_1 = self.mu_affine_1.cuda()
            self.mu_affine_2 = self.mu_affine_2.cuda()

            self.sigma_affine_1 = self.sigma_affine_1.cuda()
            self.sigma_affine_2 = self.sigma_affine_2.cuda()

            self.x_affine_1 = self.x_affine_1.cuda()
            self.x_affine_2 = self.x_affine_2.cuda()

            self.y_affine_1 = self.y_affine_1.cuda()
            self.y_affine_2 = self.y_affine_2.cuda()

            self.softmax_approx_x = self.softmax_approx_x.cuda()
            self.softmax_approx_y = self.softmax_approx_y.cuda()


        self.standard_normal = distb.MultivariateNormal(std_mean, std_cov)


    def approximate_softmax(self, x, x_neg, z, approx_embed):
        numerator = torch.exp(torch.matmul(z, approx_embed(x)))
        denominator = torch.exp(torch.matmul(z, approx_embed(x_neg).t()))
        # praise the Adam gods and forget the constant terms
        return numerator / (numerator + denominator.sum())

    def x_inference(self, x, neg_x):
        if self.use_cuda:
            x = x.cuda()
            neg_x = neg_x.cuda()

        x_embed = self.internal_embedding(x)
        neg_x_embed = self.internal_embedding(neg_x)
        hidden_op = torch.zeros(x_embed.size(0), self.embedding_dim)
        hidden_state = (torch.randn(2, 1, self.embedding_dim),
                    torch.randn(2, 1, self.embedding_dim))
        if self.use_cuda:
            hidden_op.cuda()
            hidden_state = (hidden_state[0].cuda(), hidden_state[1].cuda())

        log_xi_sum = torch.zeros(1)
        kl_sum = torch.zeros(1)

        z_i_all = torch.zeros(x.size(0), self.embedding_dim)
        mu_all = torch.zeros(x.size(0), self.embedding_dim)
        sigma_all = torch.zeros(x.size(0), self.embedding_dim)

        if self.use_cuda:
            log_xi_sum = log_xi_sum.cuda()
            kl_sum = kl_sum.cuda()

            z_i_all = z_i_all.cuda()
            mu_all = mu_all.cuda()
            sigma_all = sigma_all.cuda()


        for idx, (i, x_i) in enumerate(zip(x_embed, x)):

            out, hidden_state = self.lstm(i.view(1, 1, -1), hidden_state)
            hidden, cell_state = hidden_state[0], hidden_state[1]
            hidden_sum = hidden[0] + hidden[1]

            mu = F.relu(self.mu_affine_1(hidden_sum))
            mu = self.mu_affine_2(mu)
            mu_all[idx] = mu

            sigma = F.relu(self.sigma_affine_1(hidden_sum))
            sigma = F.softplus(self.sigma_affine_2(sigma))
            sigma_all[idx] = sigma

            z_i = mu + torch.mul(self.standard_normal.sample().detach(), sigma).squeeze()
            z_i_all[idx] = z_i

            log_xi_sum += self.approximate_softmax(x_i, neg_x, z_i, self.softmax_approx_x)

            mu_2 = Variable(torch.zeros(self.embedding_dim), requires_grad=True)
            sigma_2 = Variable(torch.ones(self.embedding_dim), requires_grad=True)
            if self.use_cuda:
                mu_2 = mu_2.cuda()
                sigma_2 = sigma_2.cuda()

            kl_loss = kl_div(mu, sigma, mu_2, sigma_2)
            kl_sum += kl_loss

        return z_i_all, mu_all, sigma_all, log_xi_sum, kl_sum

    def get_alignments(self, x, y, neg_y, z_i_all):
        if self.use_cuda:
            y = y.cuda()
            neg_y = neg_y.cuda()

        log_yi_sum = torch.zeros(1)
        alignment = torch.zeros(y.size(0))
        a_j = 1 / x.size(0)

        if self.use_cuda:
            log_yi_sum = log_yi_sum.cuda()
            alignment = alignment.cuda()

        for y_idx, y_i in enumerate(y):
            max_am, max_am_idx = -1, -1
            for a_m in range(0, x.size(0)):
                p_j_z_a_j = self.approximate_softmax(y_i, neg_y, z_i_all[a_m], self.softmax_approx_y)
                if max_am < p_j_z_a_j:
                    max_am_idx = a_m
                    max_am = p_j_z_a_j
                log_yi_sum += a_j * p_j_z_a_j

            alignment[y_idx] = max_am_idx
        return alignment, log_yi_sum

    def forward(self, x, y, neg_x, neg_y):
        z_i_all, _, _, log_xi_sum, kl_sum = self.x_inference(x, neg_x)
        alignment, log_yi_sum = self.get_alignments(x, y, neg_y, z_i_all)
        pos_loss = log_xi_sum + log_yi_sum - kl_sum

        return -1 * pos_loss


# alignments are independent of the other alignments
# there are as many z as xs
# there as many a as ys
# generate embeddings, generate english words from embeddings then generate alignments, then generate french word given alignment and embedding
# embeddings are i.i.d
# x_i depends only on the corresponding z_i
# alignment selects only one z i.e z_{a_j}
# prior z_i is standard normal
# Cat(x_i | f_{\theta}(z_i)) f_{\theta} takes d to V_x (softmax)
# alignment is uniform over (1/m) m -> number of x
# Cat(y_i | g_{\theta}(z_{a_j}))

# two neural nets in generative g and f (both MLPs)
# two more neural nets for q(z_1^m | x_1^m) -> mean and sigma
#

def to_index(sentence, vocab):
    sentence = vocab.process(sentence)
    return [vocab[word] for word in sentence], [word for word in sentence]


if __name__ == '__main__':

    ##### PARAMS ####
    params = {
        "embedding_dim" : 200,
        "vocab_x": 10000,
        "vocab_y": 10000,
        "n_epochs": 1,
        "random_state" : 42,
        "en_data_path" : "data/hansards/small_training.en",
        "fr_data_path" : "data/hansards/small_training.fr",
        "model_name": "eam_hansards_small",
        "en_stop_words_path" : None,
        "fr_stop_words_path" : None,
        "use_cuda" : True,
        "n_negative": 100,
        "stopping_loss": 1e-3
    }
    #################

    locals().update(params)

    en_stop_words, fr_stop_words = None, None
    if en_stop_words_path:
        en_stop_words = read_stop_words(en_stop_words_path)

    if fr_stop_words_path:
        fr_stop_words = read_stop_words(fr_stop_words_path)

    en_sentences = SentenceIterator(en_data_path, stop_words=en_stop_words)
    fr_sentences = SentenceIterator(fr_data_path, stop_words=fr_stop_words)
    en_vocab = Vocabulary(en_sentences, max_size = vocab_x)
    fr_vocab = Vocabulary(fr_sentences, max_size = vocab_y)

    eam = EmbedAlignModel(en_vocab, fr_vocab, embedding_dim, random_state = random_state, use_cuda = use_cuda)

    optimizer = optim.Adam(eam.parameters())

    tictoc = utils.TicToc()
    epoch_losses = []
    for epoch in np.arange(1, n_epochs + 1):
        print("Running epoch: ", epoch)
        epoch_loss = utils.Mean()
        for sentence_num, (en_sentence, fr_sentence) in enumerate(zip(en_sentences, fr_sentences)):
            if sentence_num % 10 == 0:
                tictoc.tic("Sentence: {} of {}: Mean Loss: {}".format(sentence_num + 1, en_vocab.sentence_count, epoch_loss.mean()))

            en_matrix, en_words = to_index(en_sentence, en_vocab)
            en_matrix = torch.LongTensor(en_matrix)
            fr_matrix, fr_words = to_index(fr_sentence, fr_vocab)
            fr_matrix = torch.LongTensor(fr_matrix)

            neg_samples_en = get_negative_batch(en_vocab, n_negative, en_words)
            neg_samples_fr = get_negative_batch(fr_vocab, n_negative, fr_words)

            optimizer.zero_grad()
            loss = eam(en_matrix, fr_matrix, torch.LongTensor(neg_samples_en), torch.LongTensor(neg_samples_fr))
            loss.backward()
            optimizer.step()

            if use_cuda:
                epoch_loss.add(loss.cpu().item())
            else:
                epoch_loss.add(loss.item())

            if sentence_num > 1000 and epoch_loss.mean() < stopping_loss:
                print("Loss is less than stopping criterion, breaking training loop: {}".format(epoch_loss.mean()))
                break

        epoch_losses.append(epoch_loss.mean())
        tictoc.tic("Epoch complete: Mean loss: {}".format(epoch_loss.mean()))

    eam.save_model("./models", model_name, epoch_losses, params)
    del eam

    EmbedAlignModel.load("./models", model_name)
