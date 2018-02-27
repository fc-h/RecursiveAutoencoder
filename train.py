"""
Recursive autoencoder

"""
import numpy as np

from model import RecursiveAutoencoder
from decoder import ReverseRecursiveAutoencoder
from data_operation import DataOperation


INPUT_DIM = 250
LATENT_DIM = 250
BATCH_SIZE = 10


def main():
    rae = RecursiveAutoencoder(INPUT_DIM, LATENT_DIM, "load")
    rrae = ReverseRecursiveAutoencoder(INPUT_DIM, LATENT_DIM)

    do = DataOperation()
    for _ in range(10):
        seq, seq_vec = do.get_random_seq()[::]
        print(seq)
        x = np.array([seq_vec[0]])
        y = np.array([seq_vec[1]])
        for i in range(len(seq_vec) - 1):
            data = [x, y]
            rae.train(data, BATCH_SIZE)
            y = rae.predict_at_encoder(data)
            x = np.array([seq_vec[i]])
        print(y)
        rae.save_models()


if __name__ == "__main__":
    main()
