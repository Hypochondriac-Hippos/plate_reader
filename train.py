#!/usr/bin/env python3

"""
Train license plate NNs
"""

import os

import keras

import util

IMAGE_DIR = os.path.expanduser("~/Videos/353_recordings/images")
ID_OUT = os.path.abspath("./trained/id_classifier")
PLATE_OUT = os.path.abspath("./trained/plate_classifier")


def ensure_output_dirs():
    os.makedirs(os.path.dirname(ID_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(PLATE_OUT), exist_ok=True)


def load_id_dataset(subset):
    return keras.preprocessing.image_dataset_from_directory(
        os.path.join(IMAGE_DIR, "ids/train"),
        labels="inferred",
        label_mode="categorical",
        image_size=util.image_shape[:2],
        validation_split=0.2,
        subset=subset,
        seed=3141,
    )


if __name__ == "__main__":
    ensure_output_dirs()

    ids_train = load_id_dataset("training")
    ids_validation = load_id_dataset("validation")
