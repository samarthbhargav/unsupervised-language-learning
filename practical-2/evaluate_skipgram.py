import utils
import gensim

from similarity import cosine_similarity
from lst_data import LstIterator
from skipgram import SkipGramModel

if __name__ == '__main__':
    print("Hello World")
    model_name = "skipgram"
    model_root = "./models"
    test_file = "./data/lst/lst_test.preprocessed"
    gold_file = "./data/lst/lst.gold.candidates"
    print("Loading: ", model_name)
    sgm_model, loss, params = SkipGramModel.load(model_root, model_name)
    print("Loaded. Creating embeddings")
    sgm_ematrix, sgm_words = sgm_model.get_embeddings()

    gold = LstIterator(gold_file, category="subs")
    gold_list = []
    for i in gold:
        gold_list.append(i.full_sentence)
    gold_list.append(sgm_words)

    w2v = gensim.models.Word2Vec(gold_list, min_count=1, size=200)

    gold_dict = {i.target_word: i.substitutes for i in gold}
    # sgm_dict = {sgm_words[i]: sgm_ematrix[i] for i in range(len(sgm_words))}

    temp={}

    for target_gold, gold_subs in gold_dict.items():
        for target_test in sgm_words:
            if target_test == target_gold:
                temp[target_test] = gold_subs

    print("Computing similarities")
    lst = LstIterator(test_file, category = "test")
    skipped_count = 0
    existing_words = {}

    for i, l in enumerate(lst):
    #     print("Word: ", i)
        if l.target_word not in temp.keys() or l.sentence_id in existing_words.keys() or l.target_word.isdigit():
            skipped_count += 1
            existing_words[l.complete_word, l.sentence_id] = []
            continue
        temp_list = []
        for each_subs in temp[l.target_word]:
            temp_cosine = cosine_similarity(w2v.wv[l.target_word], w2v.wv[each_subs])
            temp_list.append((each_subs, temp_cosine))
        existing_words[l.complete_word, l.sentence_id] = temp_list
        # print("Words skipped {}".format(skipped_count))
    print("Number of unique words in the file: ", len(existing_words))

    with open('skipgram_lst.out', 'w+') as f:
        for k, v in existing_words.items():
            f.write('RANKED\t' + ' '.join(k))
            for word, score in v:
                f.write('\t' + word + ' ' + str(round(score, 4)) + '\t')
            f.write('\n')
            # f.write(k + v +  '\n')
