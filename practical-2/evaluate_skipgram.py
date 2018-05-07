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
        if l.target_word not in wsm.word_index or l.sentence_id in existing_words.keys() or l.target_word.isdigit():
            skipped_count += 1
            continue
        existing_words[l.complete_word, l.sentence_id] = (wsm.most_similar(l.target_word, score=True, n=3))
        # print("Words skipped {}".format(skipped_count))
    print("Number of unique words in the file: ", len(existing_words))

with open('lst.out', 'w+') as f:
    for k, v in existing_words.items():
        f.write('RANKED\t' + ' '.join(k))
        for word, score in v:
            f.write('\t' + word + ' ' + str(round(score, 4)) + '\t')
        f.write('\n')
        # f.write(k + v +  '\n')
