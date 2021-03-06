import os
import sys
import logging
import argparse
import pickle as pkl

import torch
import gensim
import sklearn
import numpy as np
from gensim.models.word2vec import Word2Vec


PATH_TO_SENTEVAL = 'SentEval'
sys.path.insert(0, PATH_TO_SENTEVAL)

import senteval


from skipgram import SkipgramSentEval
from ea_sent_eval import EmbedAlignSentEval

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

CUDA_TASKS = ["SNLI", "STSBenchmark", "SICKRelatedness"]

if torch.cuda.is_available():
    logging.info("CUDA is available, evaluation on {} is enabled".format(CUDA_TASKS))
    TASKS = CUDA_TASKS[:]
else:
    TASKS = []
    logging.info("CUDA is not available, evaluation on {} is disabled".format(CUDA_TASKS))


# TODO: Fix "ImageCaptionRetrieval"
TASKS.extend(["STS12", "STS13",
            "STS14", "STS15", "STS16", "CR", "MR", "MPQA", "SUBJ",
            "SST2", "SST5", "TREC", "MRPC", "SICKEntailment"])

TASKS = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC', 'SICKEntailment', 'STS14']

#TASKS = ['CR', 'MR']


PROBING_TASKS = ['Length', 'WordContent', 'Depth', 'TopConstituents',
'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
'OddManOut', 'CoordinationInversion']


def get_params(args):
    params = {'task_path': args.data_path, 'usepytorch': torch.cuda.is_available(), 'kfold': args.k_fold, "seed": 42}
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

    logging.info("********** Params: {} ********".format(params))
    return params

def evaluate_model(model, args):
    se = senteval.engine.SE(get_params(args), model.batcher, model.prepare)

    results = se.eval(TASKS)

    if not os.path.exists(args.results_folder):
        os.mkdir(args.results_folder)

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
    parser.add_argument("--embed-align", dest="embed_align_location", help="Location of the folder containing the Embed Align model to evaluate", required=True)

    parser.add_argument("--data_path", help="Location of the SentEval data", default="SentEval/data")
    parser.add_argument("--k-fold", type=int, dest="k_fold", help="Number of folds", default=10)
    parser.add_argument("--probing", help="If true, evaluates models on probing tasks as well", default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.info("Arguments: {}".format(args))

    # Evaluate Skipgram
    skipgram = SkipgramSentEval(args.skipgram_location, args.composition_method)
    evaluate_model(skipgram, args)

    # Evaluate Embed Align
    ckpt_path = os.path.join(args.embed_align_location, "model.best.validation.aer.ckpt")
    tok_path = os.path.join(args.embed_align_location, "tokenizer.pickle")
    embed_align = EmbedAlignSentEval(ckpt_path, tok_path, args.composition_method)
    evaluate_model(embed_align, args)
