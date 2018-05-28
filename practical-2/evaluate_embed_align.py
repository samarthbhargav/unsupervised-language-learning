import collections

import torch
import numpy as np
import torch.distributions as distb

import utils
from utils import kl_div, save_scores
from lst_data import LstIterator
from data import get_context_words, SentenceIterator
from similarity import cosine_similarity
from embed_align import EmbedAlignModel, to_index, get_negative_batch

def save_scores(lst, scores, file_name):
    with open(file_name, 'w') as f:
        for lst_item in lst:
            f.write("RANKED\t {} {}".format(lst_item.complete_word, lst_item.sentence_id))
            for candidate, score in scores[lst_item.complete_word, lst_item.sentence_id]:
                f.write('\t' + candidate + ' ' + str(round(score, 4)) + '\t')
            f.write("\n")

def create_naacl_alignments(model, en_sentences, fr_sentences, n_negative, naacl_out):
    en_vocab = model.vocab_x
    fr_vocab = model.vocab_y

    alignments = {}
    for sentence_num, (en_sentence, fr_sentence) in enumerate(zip(en_sentences, fr_sentences)):
        en_matrix, en_words = to_index(en_sentence, en_vocab)
        en_matrix = torch.LongTensor(en_matrix)
        fr_matrix, fr_words = to_index(fr_sentence, fr_vocab)
        fr_matrix = torch.LongTensor(fr_matrix)

        neg_samples_en = torch.LongTensor(get_negative_batch(en_vocab, n_negative, en_words))
        neg_samples_fr = torch.LongTensor(get_negative_batch(fr_vocab, n_negative, fr_words))

        z_i_all, _, _, _, _ = model.x_inference(en_matrix, neg_samples_en)
        alignment, _ = model.get_alignments(en_matrix, fr_matrix, neg_samples_fr, z_i_all)

        alignments[sentence_num+1] = [(idx, val) for (idx, val) in enumerate(alignment)]

    with open(naacl_out, "w") as writer:
        for sentence_num, alignment in alignments.items():
            for (x, y) in alignment:
                writer.write("{} {} {}\n".format(sentence_num, x + 1, int(y + 1)))




if __name__ == '__main__':
    model_name = "test_ea"
    model_root = "./models"

    print("Loading: ", model_name)
    model, loss, params = EmbedAlignModel.load(model_root, model_name)
    print(params)

    alignment_files = ["data/wa/test", "data/wa/dev"]
    for alignment_file in alignment_files:
        print("Evaluation:: Alignment::", alignment_file)
        alignment_x_file = alignment_file + ".en"
        alignment_y_file = alignment_file + ".fr"
        en_sentences = SentenceIterator(alignment_x_file, stop_words=params["en_stop_words_path"])
        fr_sentences = SentenceIterator(alignment_y_file, stop_words=params["fr_stop_words_path"])
        create_naacl_alignments(model, en_sentences, fr_sentences, params["n_negative"], alignment_file.split("/")[-1] + ".naacl")

    print("Evaluation:: LST")

    test_file = "./data/lst/lst_test.preprocessed"
    gold_file = "./data/lst/lst.gold.candidates"

    gold = LstIterator(gold_file, category="subs")
    gold_dict = {i.target_word: i.substitutes for i in gold}

    lst = LstIterator(test_file, category = "test")

    skipped = 0
    skipped_incorrect_parse = 0
    scores = collections.defaultdict(list)

    en_vocab = model.vocab_x

    for lst_item in lst:
        # first compute target distributions
        center_word = lst_item.target_word
        center_idx = lst_item.target_pos

        # a sanity check
        assert (lst_item.complete_word, lst_item.sentence_id) not in scores

        if center_word not in en_vocab.index:
            print("Center Word {} not in vocab".format(center_word))
            skipped += 1
            continue

        en_matrix, en_words = to_index(lst_item.tokenized_sentence, en_vocab)
        en_matrix = torch.LongTensor(en_matrix)
        neg_samples_en = torch.LongTensor(get_negative_batch(en_vocab, params["n_negative"], en_words))

        _, mu_all, sigma_all, _, _ = model.x_inference(en_matrix, neg_samples_en)
        center_mu = mu_all[center_idx]
        center_sigma = sigma_all[center_idx]

        for gold_candidate in gold_dict[lst_item.target_word]:
            if gold_candidate not in en_vocab.index:
                # TODO print something out and track
                continue

            en_sentence = lst_item.tokenized_sentence
            en_sentence[center_idx] = gold_candidate
            en_matrix, en_words = to_index(en_sentence, en_vocab)
            en_matrix = torch.LongTensor(en_matrix)
            neg_samples_en = torch.LongTensor(get_negative_batch(en_vocab, params["n_negative"], en_words))

            _, mu_all, sigma_all, _, _ = model.x_inference(en_matrix, neg_samples_en)

            candidate_mu = mu_all[center_idx]
            candidate_sigma = sigma_all[center_idx]
            kl = -1 * kl_div(center_mu, center_sigma, candidate_mu, candidate_sigma)
            scores[lst_item.complete_word, lst_item.sentence_id].append((gold_candidate, kl.item()))

    print("Skipped: {}".format(skipped))

    save_scores(lst, scores, "eam.out")
