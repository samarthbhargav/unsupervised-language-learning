import os
import sys
import logging
import argparse
import pickle as pkl

import gensim
import sklearn
import numpy as np
from gensim.models.word2vec import Word2Vec


PATH_TO_SENTEVAL = 'SentEval'
sys.path.insert(0, PATH_TO_SENTEVAL)

import senteval


from skipgram import SkipgramSentEval

TASKS = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
           'SICKRelatedness', 'SICKEntailment', 'STSBenchmark',
           'SNLI', 'ImageCaptionRetrieval', 'STS12', 'STS13',
           'STS14', 'STS15', 'STS16']

TASKS = ['CR', 'MR']


PROBING_TASKS = ['Length', 'WordContent', 'Depth', 'TopConstituents',
'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
'OddManOut', 'CoordinationInversion']

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def get_params(args):
    return {'task_path': args.data_path,
            'usepytorch': False,
            'kfold':args.k_fold}

def evaluate_model(model, args):
    se = senteval.engine.SE(get_params(args), model.batcher, model.prepare)

    results = se.eval(TASKS)

    with open(os.path.join(args.results_folder, model.get_name() + ".pkl"), "wb") as handle:
        pkl.dump(results, handle)


    if args.probing:
        probing_results = se.eval(PROBING_TASKS)
        with open(os.path.join(args.results_folder, model.get_name() + "probing.pkl"), "wb") as handle:
            pkl.dump(probing_results, handle)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate all models')
    parser.add_argument("results_folder", help="Location of the folder to store results")
    parser.add_argument("composition_method", help="How to compose word embeddings into a sentence embedding", choices=["avg", "min", "max", "concat"])
    parser.add_argument("--skipgram", dest="skipgram_location", help="Location of the skipgram model to evaluate", required=True)
    parser.add_argument("--data_path", help="Location of the SentEval data", default="SentEval/data")
    parser.add_argument("--k-fold", dest="k_fold", help="Number of folds", default=10)
    parser.add_argument("--probing", help="If true, evaluates models on probing tasks as well", default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    skipgram = SkipgramSentEval(args.skipgram_location, args.composition_method)

    evaluate_model(skipgram, args)
