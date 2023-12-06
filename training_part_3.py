# %%

import random
import csv

import numpy
import tensorflow as tf
from keras import layers, utils, losses
from matplotlib import pyplot
from tqdm import tqdm

from model import UNet, ClassifierNet

# Check for GPU
print(tf.config.list_physical_devices("GPU"))

# For seed sync
random.seed()

RAW_WIDTH = 256
RAW_HEIGHT = 256
RAW_SIZE = (RAW_WIDTH, RAW_HEIGHT)
WIDTH = 256
HEIGHT = 256
SIZE = (WIDTH, HEIGHT)
INPUTS_DIR = "datasets/training/images/"
LABELS_FILE = "datasets/training/classification.csv"
BATCH_SIZE = 50

# %%

with open(LABELS_FILE, mode="r", encoding="utf8") as file:
    reader = csv.reader(file)
    next(reader) # skip header
    index = list(reader)
    target = numpy.array(list(map(
        lambda x: [float(x[1]) > 0.5, float(x[2]) > 0.5],
        index
    )))
    index = list(map(
        lambda x: INPUTS_DIR + x[0] + ".jpg",
        index
    ))

# %%

seg_model = UNet()
seg_model(layers.Input((WIDTH, HEIGHT, 3)))

# %%

seg_model.load_weights("saved/v1/weights.h5")

# %%

cls_model = ClassifierNet()
cls_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=losses.BinaryCrossentropy(),
    metrics=[
        "CategoricalAccuracy",
        "F1Score"
    ]
)
cls_model(layers.Input((WIDTH, HEIGHT, 3)))
cls_model.encoder.set_weights(seg_model.encoder.get_weights())

# %%

# random.shuffle(index)

def load_image(path):
    file = tf.io.read_file(path)
    image = tf.io.decode_image(file, channels=3, expand_animations=False)
    image = tf.image.resize(image, size=RAW_SIZE)
    return image

paths_ds = tf.data.Dataset.from_tensor_slices(index)

ds = paths_ds.map(load_image)
inputs_ds = ds.batch(BATCH_SIZE).cache()

labels_ds = tf.data.Dataset.from_tensor_slices(target)
labels_ds = labels_ds.batch(BATCH_SIZE).cache()

# %%

zipped = tf.data.Dataset.zip(inputs_ds, labels_ds)
for (inputs, labels) in zipped:
    cls_model.train(inputs, labels)
