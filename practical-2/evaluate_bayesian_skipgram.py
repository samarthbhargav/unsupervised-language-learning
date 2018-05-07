import utils

from lst_data import LstIterator, LstItem
from bayesian_skipgram import BayesianSkipgram

def compute_score(lst_item, method="mean"):
    pass


if __name__ == '__main__':
    model_name = "test_bn"
    model_root = "./models"
    filename = "./data/lst/lst_test.preprocessed"
    model, loss, params = BayesianSkipgram.load(model_root, model_name)

    vocab = model.vocab

    lst = LstIterator(filename)
    skipped_count = 0
    existing_words = {}
    for l in lst:
        print(l)

        # print("Words skipped {}".format(skipped_count))
    print("Number of unique words in the file: ", len(existing_words))

    for k, v in existing_words.items():
        # with open('output', 'w') as f:
        #     f.append(k + "\t" + "\n")
        print(k, v)
