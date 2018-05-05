import utils

from similarity import WordSimilarityModel
from lst_data import LstIterator, LstItem
from skipgram import SkipGramModel

if __name__ == '__main__':
    model_name = "test"
    model_root = "./models"
    filename = "./data/lst/lst_test.preprocessed"
    model, loss, params = SkipGramModel.load(model_root, model_name)
    ematrix, words = model.get_embeddings()

    vocab = model.vocab

    wsm = WordSimilarityModel(words, ematrix)
    lst = LstIterator(filename)
    skipped_count = 0
    existing_words = {}
    for l in lst:
        if l.target_word not in wsm.word_index or l.target_word in existing_words.keys() or l.target_word.isdigit():
            skipped_count += 1
            continue
        existing_words[l.target_word] = (wsm.most_similar(l.target_word, score=True, n=3))
        # print("Words skipped {}".format(skipped_count))
    print("Number of unique words in the file: ", len(existing_words))

    for k, v in existing_words.items():
        # with open('output', 'w') as f:
        #     f.append(k + "\t" + "\n")
        print(k, v)
