#!/usr/bin/env python3

"""
Train license plate NNs
"""

import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import models
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


def visualize_dataset(dataset):
    """Preview a few images and labels from a set to make sure everything is in order."""
    plt.figure()
    for images, labels in dataset.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(dataset.class_names[np.where(labels[i])[0][0]])
            plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ensure_output_dirs()

    ids_train = load_id_dataset("training")
    ids_validation = load_id_dataset("validation")

    visualize_dataset(ids_train)
    visualize_dataset(ids_validation)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ids_train = ids_train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    ids_validation = ids_validation.cache().prefetch(buffer_size=AUTOTUNE)

    ids = models.PlateID(util.image_shape)
    ids.model.summary()
    ids.model.fit(ids_train, validation_data=ids_validation, epochs=5)
