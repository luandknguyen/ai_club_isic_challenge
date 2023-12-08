import os
import shutil

SIGNATURE = "_superpixels.png"

def move_superpixels_images(from_dir, to_dir):
    for image in os.listdir(from_dir):
        if SIGNATURE in image:
            old_file = from_dir + image
            new_file = to_dir + image
            shutil.move(old_file, new_file)
            print(f"Moving {image} to new dir")

