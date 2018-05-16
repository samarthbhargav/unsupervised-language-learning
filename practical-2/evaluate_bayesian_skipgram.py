import collections

import torch
import numpy as np
import torch.distributions as distb

import utils
from utils import kl_div, save_scores
from lst_data import LstIterator
from data import get_context_words
from similarity import cosine_similarity
from bayesian_skipgram import BayesianSkipgram

if __name__ == '__main__':
    model_name = "bsg_small_en"
    model_root = "./models"
    test_file = "./data/lst/lst_test.preprocessed"
    gold_file = "./data/lst/lst.gold.candidates"
    print("Loading: ", model_name)
    model, loss, params = BayesianSkipgram.load(model_root, model_name)
    context_window = params["context_window"]
    vocab = model.vocab

    gold = LstIterator(gold_file, category="subs")
    gold_dict = {i.target_word: i.substitutes for i in gold}

    lst = LstIterator(test_file, category = "test")

    skipped = 0
    skipped_incorrect_parse = 0
    scores_mu = collections.defaultdict(list)
    scores_kl_post = collections.defaultdict(list)
    scores_kl_prior = collections.defaultdict(list)

    for lst_item in lst:
        # first compute target distributions
        center_word = lst_item.target_word
        center_idx = lst_item.target_pos

        # a sanity check
        assert (lst_item.complete_word, lst_item.sentence_id) not in scores_mu

        if center_word not in vocab.index:
            print("Center Word {} not in vocab".format(center_word))
            skipped += 1
            continue



        context_words = get_context_words(vocab.process(lst_item.tokenized_sentence), center_idx, context_window, pad=True)
        context_vec = np.zeros(len(context_words))
        for i, context_word in enumerate(context_words):
            context_vec[i] = vocab[context_word]


        center_vec = torch.LongTensor(np.array([vocab[center_word]]))
        context_vec = torch.LongTensor(context_vec)
        if model.use_cuda:
            center_vec = center_vec.cuda()
            context_vec = context_vec.cuda()

        mu, sigma, inf_mu, inf_sigma, z = model.get_distribution(center_vec, context_vec)

        #print(mu, sigma, inf_mu, inf_sigma, z)
        for gold_candidate in gold_dict[lst_item.target_word]:
            if gold_candidate not in vocab.index:
                # TODO print something out and track
                continue
            vec = torch.LongTensor(np.array([vocab[gold_candidate]]))

            if model.use_cuda:
                center_vec = center_vec.cuda()

            mu_s, sigma_s, inf_mu_s, inf_sigma_s, z_s = model.get_distribution(vec, context_vec)
            score_mu = cosine_similarity(mu.squeeze().detach().cpu().numpy(), mu_s.squeeze().detach().cpu().numpy())
            scores_mu[lst_item.complete_word, lst_item.sentence_id].append((gold_candidate, score_mu))

            # TODO not sure if minus is reqd
            # posterior (word) (inf) || prior (candidate)
            kl_prior = -1 * kl_div(inf_mu, inf_sigma, mu_s, sigma_s)
            scores_kl_prior[lst_item.complete_word, lst_item.sentence_id].append((gold_candidate, kl_prior.item()))

            kl_post = -1 * kl_div(inf_mu, inf_sigma, inf_mu_s, inf_sigma_s)
            scores_kl_post[lst_item.complete_word, lst_item.sentence_id].append((gold_candidate, kl_post.item()))

    print("Skipped: {}".format(skipped))

    save_scores(lst, scores_mu, "bs_mu_lst.out")
    save_scores(lst, scores_kl_post, "bs_kl_post_lst.out")
    save_scores(lst, scores_kl_prior, "bs_kl_prior_lst_mu.out")
