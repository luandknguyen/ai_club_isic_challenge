# %%

import random
import gc

import numpy
import tensorflow as tf
from keras import layers, utils, losses
from matplotlib import pyplot
from tqdm import tqdm

from model import UNet

# Check for GPU
print(tf.config.list_physical_devices("GPU"))

# For seed sync
random.seed()

RAW_WIDTH = 512
RAW_HEIGHT = 512
RAW_SIZE = (RAW_WIDTH, RAW_HEIGHT)
WIDTH = 512
HEIGHT = 512
SIZE = (WIDTH, HEIGHT)
INPUTS_DIR = "datasets/training/images/"
LABELS_DIR = "datasets/training/segmentation/"
BATCH_SIZE = 20

# %% Load dataset

seed = random.randint(0, 2**32 - 1)

input_images = utils.image_dataset_from_directory(
    INPUTS_DIR,
    labels=None,
    label_mode=None,
    image_size=RAW_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    interpolation="bilinear",
    seed=seed
)

input_images = input_images.cache()

label_images = utils.image_dataset_from_directory(
    LABELS_DIR,
    labels=None,
    label_mode=None,
    image_size=RAW_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    interpolation="nearest",
    seed=seed
)

label_images = label_images.cache()

# %%


model = UNet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=losses.BinaryCrossentropy(),
    metrics=[
        "BinaryAccuracy",
        "BinaryIoU"
    ]
)

# %%


class Preprocessing():
    def __init__(self):
        seed = random.randint(0, 2**32 - 1)
        self.scale = layers.Rescaling(scale=1./255)
        self.rotate_x = layers.RandomRotation(
            factor=0.2, fill_mode="constant", interpolation="bilinear", seed=seed)
        self.rotate_y = layers.RandomRotation(
            factor=0.2, fill_mode="constant", interpolation="nearest", seed=seed)
        self.crop_x = layers.RandomCrop(height=HEIGHT, width=WIDTH, seed=seed)
        self.crop_y = layers.RandomCrop(height=HEIGHT, width=WIDTH, seed=seed)
        self.brightness = layers.RandomBrightness(
            factor=0.2, value_range=(0, 1))
        self.contrast = layers.RandomContrast(factor=0.2)
        self.flip_x = layers.RandomFlip(seed=seed)
        self.flip_y = layers.RandomFlip(seed=seed)

    def __call__(self, inputs, labels):
        index = numpy.arange(inputs.shape[0] * 2)
        numpy.random.shuffle(index)
        # Inputs
        x = self.scale(inputs)
        x = self.rotate_x(x)
        #x = self.crop_x(x)
        x = tf.repeat(x, repeats=2, axis=0)
        x = self.brightness(x)
        x = self.contrast(x)
        x = self.flip_x(x)
        x = tf.convert_to_tensor(x.numpy()[index])
        # Labels
        y = self.scale(labels)
        y = self.rotate_y(y)
        #y = self.crop_y(y)
        y = tf.repeat(y, repeats=2, axis=0)
        y = self.flip_y(y)
        y = tf.convert_to_tensor(y.numpy()[index], dtype="int8")
        return (x, y)


preprocessing = Preprocessing()

# %%

EPOCH = 10
iou_history = []

for epoch in range(EPOCH):
    print(f"=== Epoch {epoch} ===")
    zipped = tf.data.Dataset.zip((input_images, label_images))
    for (inputs, labels) in tqdm(zipped):
        inputs, labels = preprocessing(inputs, labels)
        history = model.fit(inputs, labels, verbose=0)
        iou_history.append(history.history["binary_io_u"])
    gc.collect()

# %%

model.save_weights("saved/v1/weights.h5", save_format="h5")
