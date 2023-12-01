import os
import shutil

INPUT_DIR = "datasets/training/images/"
OUTPUT_DIR = "datasets/training/superpixels/"
SIGNATURE = "_superpixels.png"

for image in os.listdir(INPUT_DIR):
    if SIGNATURE in image:
        old_file = INPUT_DIR + image
        new_file = OUTPUT_DIR + image
        shutil.move(old_file, new_file)
        print(f"Moving {image} to new dir")

