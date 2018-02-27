
from keras.layers import Input, Dense, merge, concatenate
from keras.models import Model
from keras.optimizers import RMSprop

# from keras.layers import Lambda, Input, Dense, GRU, LSTM, RepeatVector, concatenate, Dropout, Bidirectional
# from keras.layers.core import Flatten, Reshape
# from keras.layers import merge, multiply
# from keras.optimizers import Adam, SGD, RMSprop


class RecursiveAutoencoder():
    def __init__(self, inout_dim, latent_dim, flag=""):
        self.input_dim = inout_dim
        self.latent_dim = latent_dim
        self.output_dim = inout_dim
        self.encoder_save_dir = "./wait/encoder.hdf5"
        self.decoder_save_dir = "./wait/decoder.hdf5"
        if flag == "load":
            self.encoder = self.load_models(self.encoder_save_dir)
            self.decoder = self.load_models(self.decoder_save_dir)
        else:
            self.encoder = self.build_encoder()
            self.decoder = self.build_decoder()
        self.build_autoencoder()

    def build_encoder(self):
        input_layer1 = Input(shape=(self.input_dim,))
        input_layer2 = Input(shape=(self.input_dim,))
        merged_layer = concatenate([input_layer1, input_layer2])
        latent_layer = Dense(
            self.latent_dim, activation="sigmoid")(merged_layer)
        return Model([input_layer1, input_layer2], latent_layer)

    def build_decoder(self):
        input_layer = Input(shape=(self.latent_dim,))
        output_layer1 = Dense(
            self.latent_dim, activation="linear")(input_layer)
        output_layer2 = Dense(
            self.latent_dim, activation="linear")(input_layer)
        return Model(input_layer, [output_layer1, output_layer2])

    def build_autoencoder(self):
        _, _, mearged, latent = self.encoder.layers
        input_layer1 = Input(shape=(self.input_dim,))
        input_layer2 = Input(shape=(self.input_dim,))
        merged_layer = concatenate([input_layer1, input_layer2])
        latent_layer = latent(merged_layer)

        _, out1, out2 = self.decoder.layers
        output_layer1 = out1(latent_layer)
        output_layer2 = out2(latent_layer)

        self.autoencoder = Model([input_layer1, input_layer2], [
            output_layer1, output_layer2])

        optimizer = RMSprop(lr=0.001, rho=0.7, epsilon=1e-08, decay=0.0)

        self.autoencoder.compile(optimizer=optimizer,
                                 loss='mean_squared_error',
                                 metrics=['accuracy'])
        self.autoencoder.summary()

    def train(self, data, batch_size):
        loss = self.autoencoder.fit([data[0], data[1]], [data[0], data[1]],
                                    batch_size=batch_size, epochs=10)
        return loss

    def predict_at_encoder(self, data):
        return self.encoder.predict([data[0], data[1]])

    def predict_autoencoder(self, data):
        return self.autoencoder.predict([data[0], data[1]])

    def save_models(self):
        print("save model")
        self.encoder.save(self.encoder_save_dir)
        self.decoder.save(self.decoder_save_dir)

    def load_models(self, fname):
        print("load " + fname)
        from keras.models import load_model
        return load_model(fname)
