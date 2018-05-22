#from __future__ import absolute_import, division, unicode_literals



import os
import sys
import logging
import argparse


import gensim
from gensim.models.word2vec import Word2Vec

import numpy as np
import sklearn

# Load code from previous practical
sys.path.insert(0, "../practical-2/")
import data

PATH_TO_SENTEVAL = 'SentEval'
sys.path.insert(0, PATH_TO_SENTEVAL)

import senteval


from api import SentEvalApi


logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
EUROPARL_DATA = "../practical-2/data/hansards/training.en"


def get_skipgram(location):
    if os.path.exists(location):
        logging.info("Model already exists, loading it!")
        model = Word2Vec.load(location)
    else:
        logging.info("Creating the model")
        model = Word2Vec(data.SentenceIterator(EUROPARL_DATA),
                         min_count=0,
                         size=300,
                         window=5,
                         sample=0.001,
                         sg=1, # We want to use Skipgram
                         negative = 20,
                         compute_loss=True,
                         workers=4,
                         iter=10
                        )
        logging.info("Done, saving it to {}".format(location))
        model.save(location)
    return model


class SkipgramSentEval(SentEvalApi):

    def __init__(self, location, composition_method):
        super().__init__("skipgram")
        self.model = get_skipgram(location)
        self.composition_method = composition_method

    def prepare(self, params, samples):
        params.skipgram = self.model
        params.composition_method = self.composition_method

    def batcher(self, params, batch):
        # if a sentence is empty dot is set to be the only token
        # you can change it into NULL dependening in your model
        batch = [sent if sent != [] else ['.'] for sent in batch]
        embeddings = []

        for sent in batch:
            sentvec = []
            # the format of a sentence is a lists of words (tokenized and lowercased)
            for word in sent:
                if word in params.skipgram.wv:
                    # [number of words, embedding dimensionality]
                    sentvec.append(params.skipgram[word])
            if not sentvec:
                vec = np.zeros(params.skipgram.vector_size)
                sentvec.append(vec)
            sentvec = np.mean(sentvec, 0)
            embeddings.append(sentvec)
        embeddings = np.vstack(embeddings)
        return embeddings

def parse_args():
    parser = argparse.ArgumentParser(description='Create Skipgram model')
    parser.add_argument("location", help="Location of the model (will be created if not sepecified)")

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    model = get_skipgram(args.location)
