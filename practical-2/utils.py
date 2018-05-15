import os

import time
import pickle as pkl

import torch

class TicToc:

    def __init__(self):
        self.start_time = time.time()

    def tic(self, prefix=None):
        if prefix is None:
            prefix = ""
        else:
            prefix = "{}:: ".format(prefix)
        print("{}Time elapsed: {} seconds".format(prefix, round(time.time() - self.start_time, 2)))


def mkdir(path):
    if os.path.exists(path):
        return
    os.mkdir(path)

def save_object(object, path):
    with open(path, "wb") as writer:
        pkl.dump(object, writer)

def load_object(path):
    with open(path, "rb") as reader:
        return pkl.load(reader)

class Mean:
    def __init__(self):
        self.n = 0
        self.sum = 0

    def add(self, val):
        self.n += 1
        self.sum += val

    def mean(self):
        if self.n == 0:
            return 0
        return self.sum / self.n

def kl_div(mu_1, sigma_1, mu_2, sigma_2):
    div = torch.log(sigma_2) - torch.log(sigma_1) + (sigma_1.pow(2) + (mu_1 - mu_2).pow(2)) / (2*sigma_2.pow(2)) - 0.5
    return div.sum()


def save_scores(lst, scores, file_name):
    with open(file_name, 'w') as f:
        for lst_item in lst:
            f.write("RANKED\t {} {}".format(lst_item.complete_word, lst_item.sentence_id))
            for candidate, score in scores[lst_item.complete_word, lst_item.sentence_id]:
                f.write('\t' + candidate + ' ' + str(round(score, 4)) + '\t')
            f.write("\n")
