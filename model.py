# %% Imports

import tensorflow as tf
from keras import layers

# Check for GPU
print(tf.config.list_physical_devices("GPU"))

# %% Model


class EncoderBlock(layers.Layer):
    def __init__(self, n_filters, max_pooling=True):
        super().__init__()
        self.max_pooling = max_pooling
        self.conv_1 = layers.Conv2D(
            n_filters,
            kernel_size=3,
            activation='relu',
            padding='same'
        )
        self.conv_2 = layers.Conv2D(
            n_filters,
            kernel_size=3,
            activation='relu',
            padding='same'
        )
        if max_pooling:
            self.max_pool = layers.MaxPool2D(
                pool_size=(2, 2), strides=2, padding='same')
            
    def call(self, inputs, *args, **kwargs):
        matrix = self.conv_1(inputs)
        matrix = self.conv_2(matrix)
        skip = matrix
        if self.max_pooling:
            matrix = self.max_pool(matrix)
        return matrix, skip


class DecoderBlock(layers.Layer):
    def __init__(self, n_filters):
        super().__init__()
        self.up_sampling = layers.UpSampling2D(size=(2, 2))
        self.concat = layers.Concatenate(axis=3)
        self.conv_1 = layers.Conv2DTranspose(
            n_filters,
            kernel_size=3,
            activation='relu',
            padding='same'
        )
        self.conv_2 = layers.Conv2DTranspose(
            n_filters,
            kernel_size=3,
            activation='relu',
            padding='same'
        )
        
    def call(self, inputs, *args, **kwargs):
        inputs, skip = inputs
        matrix = self.up_sampling(inputs)
        matrix = self.concat([matrix, skip])
        matrix = self.conv_1(matrix)
        matrix = self.conv_2(matrix)
        return matrix


class Encoder(layers.Layer):
    def __init__(self):
        super().__init__()
        self.block_1 = EncoderBlock(8)
        self.block_2 = EncoderBlock(12)
        self.block_3 = EncoderBlock(16)
        self.block_4 = EncoderBlock(32, max_pooling=False)
        
    def call(self, inputs, *args, **kwargs):
        matrix, skip_1 = self.block_1(inputs)
        matrix, skip_2 = self.block_2(matrix)
        matrix, skip_3 = self.block_3(matrix)
        matrix, _ = self.block_4(matrix)
        return matrix, skip_1, skip_2, skip_3


class Decoder(layers.Layer):
    def __init__(self):
        super().__init__()
        self.block_1 = DecoderBlock(16)
        self.block_2 = DecoderBlock(12)
        self.block_3 = DecoderBlock(8)
        self.conv_out = layers.Conv2DTranspose(
            1, kernel_size=3, activation='sigmoid', padding='same')

    def call(self, inputs, *args, **kwargs):
        encoded, skip_1, skip_2, skip_3 = inputs
        matrix = self.block_1((encoded, skip_3))
        matrix = self.block_2((matrix, skip_2))
        matrix = self.block_3((matrix, skip_1))
        matrix = self.conv_out(matrix)
        return matrix


class UNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, inputs, training=False, mask=None):
        x = self.encoder(inputs)
        outputs = self.decoder(x)
        return outputs


class ClassifierNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.flatten = layers.Flatten()
        self.dense_1 = layers.Dense(20)
        self.dense_2 = layers.Dense(2)
        
    def call(self, inputs, training=False, mask=None):
        x = self.encoder(inputs)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x
        
