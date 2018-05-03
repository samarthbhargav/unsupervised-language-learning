import utils

from similarity import WordSimilarityModel

from skipgram import SkipGramModel

if __name__ == '__main__':
    model_name = "test"
    model_root = "./models"

    model, loss, params = SkipGramModel.load(model_root, model_name)
    vocab = model.vocab

    words = list(vocab.index.items())
    words.sort(key= lambda _: _[1])

    words = [ w for (w, i) in words ]
    wsm = WordSimilarityModel(words, model.input_embeddings.detach().numpy().T)
    print(wsm.most_similar("borrowing", score=True))
