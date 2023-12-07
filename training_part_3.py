# %%

import random
import gc

import numpy
import pandas
import tensorflow as tf
from keras import layers, utils, losses, metrics
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
META_FILE = "datasets/training/metadata.csv"
BATCH_SIZE = 50
SAVED_FILE = f"saved/v1/weights_{WIDTH}_{HEIGHT}.h5"
SAVED_FILE_2 = f"saved/p3/weights_{WIDTH}_{HEIGHT}.h5"

# %%

target = pandas.read_csv(LABELS_FILE, header=0)
metadata = pandas.read_csv(META_FILE, header=0)
data = pandas.merge(target, metadata, on="image_id")
data['age_approximate'] = data['age_approximate'].map(lambda x: float(x) if x != 'unknown' else -1.0)
data['male'] = (data['sex'] == 'male').astype(float)
data['female'] = (data['sex'] == 'female').astype(float)
metadata = data[['age_approximate', 'male', 'female']].to_numpy()
target = data[['melanoma', 'seborrheic_keratosis']].to_numpy()
images = (INPUTS_DIR + data['image_id'] + '.jpg').to_list()

# %%

seg_model = UNet()
seg_model(layers.Input((WIDTH, HEIGHT, 3)))
seg_model.load_weights(SAVED_FILE)

# %%

cls_model = ClassifierNet()
cls_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=losses.BinaryCrossentropy(),
    metrics=[
        "CategoricalAccuracy",
        "FalseNegatives"
    ]
)
cls_model((layers.Input((WIDTH, HEIGHT, 3)), layers.Input(3)))
cls_model.encoder.set_weights(seg_model.encoder.get_weights())

# %%

seed = random.randint(0, 2**32 - 1)

def load_image(path):
    file = tf.io.read_file(path)
    image = tf.io.decode_image(file, channels=3, expand_animations=False)
    image = tf.image.resize(image, size=RAW_SIZE)
    return image

paths_ds = tf.data.Dataset.from_tensor_slices(images)
inputs_ds = paths_ds.map(load_image)
inputs_ds = inputs_ds.shuffle(BATCH_SIZE * 8, seed=seed).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).cache()

labels_ds = tf.data.Dataset.from_tensor_slices(target)
labels_ds = labels_ds.shuffle(BATCH_SIZE * 8, seed=seed).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).cache()

metadata_ds = tf.data.Dataset.from_tensor_slices(metadata)
metadata_ds = metadata_ds.shuffle(BATCH_SIZE * 8, seed=seed).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).cache()

# %%

EPOCH = 25
loss_history = []

for epoch in range(EPOCH):
    print(f"=== Epoch {epoch} ===")
    zipped = tf.data.Dataset.zip((inputs_ds, labels_ds, metadata_ds)).shuffle(BATCH_SIZE * 2)
    for (inputs, labels, metadata) in tqdm(zipped):
        inputs = inputs / 255
        history = cls_model.fit((inputs, metadata), labels, verbose=0)
        loss_history.append(history.history["loss"])
    gc.collect()

# %%

cls_model.save_weights(SAVED_FILE_2, save_format="h5")
