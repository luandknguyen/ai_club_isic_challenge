# %%

import random

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

input_images = utils.image_dataset_from_directory(
    INPUTS_DIR,
    labels=None,
    label_mode=None,
    image_size=RAW_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    interpolation="bilinear"
)

# %%

seg_model = UNet()
seg_model(layers.Input((WIDTH, HEIGHT, 3)))

# %%

seg_model.load_weights("saved/v1/weights.h5")

# %%

cls_model = ClassifierNet()
cls_model.encoder.set_weights(seg_model.encoder.get_weights())
cls_model(layers.Input((WIDTH, HEIGHT, 3)))
