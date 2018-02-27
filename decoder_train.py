"""
Recursive autoencoder

"""
import numpy as np
from decoder import ReverseRecursiveAutoencoder
from model import RecursiveAutoencoder
from data_operation import DataOperation


INPUT_DIM = 250
LATENT_DIM = 250
BATCH_SIZE = 10


def main():
    rae = RecursiveAutoencoder(INPUT_DIM, LATENT_DIM)
    rrae = ReverseRecursiveAutoencoder(INPUT_DIM, LATENT_DIM)

    do = DataOperation()
    for _ in range(10):
        seq1, seq_vec1, seq2, seq_vec2 = do.get_random_seq2()
        print(seq1)
        x = np.array([seq_vec1[0]])
        y = np.array([seq_vec2[1]])
        for i in range(len(seq_vec1) - 1):
            data = [x, y]
            x = rae.predict_autoencoder(data)[1]
            y = rae.predict_at_encoder(data)

        sentens_vec = y
        seq_vec2 = seq_vec2[::]
        for i in range(len(seq_vec2) - 1):
            s = seq_vec2[i]
            rrae.train(y, s, BATCH_SIZE)
            y = rae.predict_at_encoder(s)


if __name__ == "__main__":
    main()
