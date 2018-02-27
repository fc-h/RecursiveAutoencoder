# 単語をベクトル化p

import gensim
import numpy as np
import sys

WAIT_FILE = "./wait/w2v.model"


class W2V():
    word_feat_len = 250

    @staticmethod
    def train(fname, saveflag="save"):
        print("train word2vec")
        sentences = gensim.models.word2vec.Text8Corpus(fname)
        #model = gensim.models.word2vec.Word2Vec(sentences, size=200, window=5, workers=4, min_count=5)
        model = gensim.models.word2vec.Word2Vec(
            sentences, size=W2V.word_feat_len, window=5, workers=4, min_count=1, hs=1)
        if saveflag == "save":
            print("save " + WAIT_FILE)
            model.save(WAIT_FILE)

    @staticmethod
    def load_model():
        # 読み込み
        print("load " + WAIT_FILE)
        model = gensim.models.word2vec.Word2Vec.load(WAIT_FILE)
        return model

    @staticmethod
    def vec_to_word(model, vec):
        return model.most_similar([vec], [], 1)[0][0]

    @staticmethod
    def vec_to_some_word(model, vec, num):
        return model.most_similar([vec], [], num)

    @staticmethod
    def str_to_vector(model, st):
        return model.wv[st]


def plot(vec):
    t = range(len(vec))
    plt.plot(t, vec)
    plt.show()


def main():
    if sys.argv[-1] == "-train":
        W2V().train("./aozora_text/files/tmp.txt", "save")
    elif sys.argv[-1] == "-load":
        W2V().load_model()
    else:
        print("set arguments -train or -load")


if __name__ == "__main__":
    main()
