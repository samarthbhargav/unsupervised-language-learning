import os
import codecs

class LST_Item:
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with codecs.open(self.filename, 'r') as f:
            for line in f:
                line = line.split()
                yield(line)

class LST_Test:
    def __init__(self, sentences):
        self.sentences = sentences

    def tokenized_sentence(self):
        return self.sentences[3:]

    def target_pos(self):
        return self.sentences[2]

    def sentence_id(self):
        return self.sentences[1]

    def target_word(self):
        return self.sentences[0]

    # def __len__(self):
    #     return len(self.sentence)

if __name__=='__main__':

    def readLST(filename):
        sentences =  LST_Item(filename)
        objList = []
        for each_sentence in sentences:
            test_obj = LST_Test(each_sentence)
            # print(test_obj.target_word())
            objList.append(test_obj)
        return objList

    dummy = readLST('data/lst/lst_test.preprocessed')
    print(dummy[10].tokenized_sentence())
