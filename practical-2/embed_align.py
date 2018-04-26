## The Embed-Align model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distb
from torch.autograd import Variable


import numpy as np

class EmbedAlign(nn.Module):
    def __init__(self, vocab_x, vocab_y, z_embedding_dim):
        super(EmbedAlign, self).__init__()

        self.vocab_x = vocab_x
        self.vocab_y = vocab_y

        self.z_embedding_dim = z_embedding_dim
        self.z = distb.MultivariateNormal(torch.zeros(self.z_embedding_dim), torch.diag(1))


    def forward(self, x):
        pass




if __name__ == '__main__':
    print(torch.diag([1, 2, 3]))
    sentences = SentenceIterator("data/hansards/training.en")
    vocab_en = Vocabulary(sentences)
    sentences = SentenceIterator("data/hansards/training.fr")
    vocab_fr = Vocabulary(sentences)

    ea = EmbedAlign(vocab_en, vocab_fr)

    ea.forward(None)
