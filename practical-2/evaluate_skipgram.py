import utils

from skipgram import SkipGramModel

if __name__ == '__main__':
    model_name = "test"
    model_root = "./models"

    model, loss, params = SkipGramModel.load(model_root, model_name)
    vocab = model.vocab

    for word in model.vocab.keys():
        print(word)
    print(model)
    print(loss)
    print(params)
