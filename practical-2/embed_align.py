import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distb
from torch.autograd import Variable

import numpy as np

from data import SentenceIterator, Vocabulary, get_context_words
import utils

class EmbedAlignModel(nn.Module):
    def __init__(self, cuda=False, random_state=None):
        super(EmbedAlignModel, self).__init__()

        if random_state:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        pass

    def forward(self, center_word, context_words):
        pass


if __name__ == '__main__':

    ##### PARAMS ####
    embedding_dim = 10
    z_embedding_dim = 15
    vocab_size = 1000
    context_window = 5
    negative_words = 10
    n_epochs = 10
    random_state = 42
    data_path = "data/wa/test.en"
    model_name = "test"
    use_cuda = True
    #################


    sentences = SentenceIterator(data_path)
    vocab = Vocabulary(sentences, max_size = vocab_size)

    bsm = BayesianSkipGramModel(vocab, embedding_dim, z_embedding_dim, random_state=random_state, cuda=use_cuda)

    # set up model parameters to learn
    model_params = [bsm.inference_embedding, bsm.prior_mean, bsm.prior_sigma]
    model_params.extend(bsm.inference_mean_transform.parameters())
    model_params.extend(bsm.inference_sigma_transform.parameters())
    model_params.extend(bsm.inference_transform.parameters())
    optimizer = optim.Adam(model_params)

    epoch_losses = []
    tictoc = utils.TicToc()
    for epoch in range(1, n_epochs + 1):
        print("Running epoch: ", epoch)
        epoch_loss = []
        for sentence_num, sentence in enumerate(sentences):
            if sentence_num % 100 == 0:
                tictoc.tic("Sentence: {}".format(sentence_num))

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

                loss = compute_loss(f_theta, z_mean, z_sigma, prior_mean, prior_sigma, context_idx, use_cuda)
                if use_cuda:
                    val = idx.cpu().data.numpy()
                    epoch_loss.append(loss.cpu().data.numpy()[0])
                else:
                    val = idx.data.numpy()
                    epoch_loss.append(loss.data.numpy()[0])
                if epoch % 5 == 0:
                    print(center_word, vocab.word(int(val)))

                loss.backward()
                optimizer.step()

        epoch_losses.append(np.mean(epoch_loss))
        tictoc.tic("Epoch complete: Mean loss: {}".format(np.mean(epoch_loss)))


    model_save_path = os.path.join("./models/", model_name)
    print("Saving model: ", model_save_path)
    torch.save(bsm.state_dict(), model_save_path)
    utils.save_object(epoch_losses, os.path.join("./models/", model_name) + "_loss")
