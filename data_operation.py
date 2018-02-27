import glob
from show_dial import ShowDial
import random
from w2v import W2V
import numpy as np


class DataOperation():
    def __init__(self):
        self.wordlists = self.get_word_lists(
            "./aozora_text/files/tmp.txt")
        self.w2v_model = W2V.load_model()

    def get_word_lists(self, file_path):
        print("make wordlists")
        # lines = open(file_path).read().split("。")
        lines = open(file_path).read().split("\n")
        wordlists = []
        for line in lines:
            wordlists.append(line.split(" "))

        print("wordlist num:", len(wordlists))
        return wordlists[:-1]

    def get_random_seq(self):
        rand = random.randint(0, len(self.wordlists) - 1)
        seq_vec = []
        seq = self.wordlists[rand]
        while('' in seq):
            seq.remove('')
        for word in seq:
            seq_vec.append(W2V().str_to_vector(self.w2v_model, word))
        return seq, seq_vec

    def get_random_seq2(self):
        rand = random.randint(0, len(self.wordlists) - 2)
        seq_vec1 = []
        seq_vec2 = []
        seq1 = self.wordlists[rand]
        seq2 = self.wordlists[rand + 1]
        while('' in seq1):
            seq1.remove('')
        while('' in seq2):
            seq2.remove('')

        for word in seq1:
            seq_vec1.append(W2V().str_to_vector(self.w2v_model, word))
        for word in seq2:
            seq_vec2.append(W2V().str_to_vector(self.w2v_model, word))
        return seq1, seq_vec1, seq2, seq_vec2

    def gen_data(self):
        seq_batch = []
        rand = random.randint(0, len(self.wordlists) - 1)
        seq = []
        while('' in self.wordlists[rand]):
            self.wordlists[rand].remove('')
        for word in self.wordlists[rand]:
            seq.append(W2V().str_to_vector(self.w2v_model, word))
            print(len(W2V().str_to_vector(self.w2v_model, word)))
        seq_batch.append(seq)
        seq_batch = np.array(seq_batch)
        print("train shape:", seq_batch.shape)
        return seq_batch

    def gen_data_batch(self, batch_size):
        seq_batch = []
        for _ in range(batch_size):
            rand = random.randint(0, len(self.wordlists) - 1)
            seq = []
            while('' in self.wordlists[rand]):
                self.wordlists[rand].remove('')
            for word in self.wordlists[rand]:
                seq.append(W2V().str_to_vector(self.w2v_model, word))
                print(len(W2V().str_to_vector(self.w2v_model, word)))
            seq_batch.append(seq)
        print("")
        print(len(seq_batch))
        print(len(seq_batch[0]))
        seq_batch = np.array(seq_batch)
        print("train shape:", seq_batch.shape)
        return seq_batch
