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
