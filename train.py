#!/usr/bin/env python

"""
Train license plate NNs
"""

import argparse
import itertools as it
import os

import matplotlib.pyplot as plt
import numpy as np

import loader
import models
import util

IMAGE_DIR = os.path.expanduser("~/Videos/353_recordings/images")
ID_OUT = os.path.abspath("./trained/id_classifier")
PLATE_OUT = os.path.abspath("./trained/plate_classifier")


def ensure_output_dirs():
    util.makedirs(os.path.dirname(ID_OUT), exist_ok=True)
    util.makedirs(os.path.dirname(PLATE_OUT), exist_ok=True)


def visualize_dataset(frames, labels):
    """Preview a few images and labels from a set to make sure everything is in order."""
    fig, axes = plt.subplots(nrows=3, ncols=3)
    axes = axes.reshape((9,))
    for i, (image, label) in enumerate(it.izip(frames[:9], labels[:9])):
        axes[i].imshow(image)
        axes[i].set_title(np.where(label)[0][0])
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Visualize datasets")
    args = parser.parse_args()

    ensure_output_dirs()

    ids_frames, ids_labels = loader.load_id_dataset(
        os.path.join(IMAGE_DIR, "ids", "train"), 0.01
    )

    if args.visualize:
        visualize_dataset(ids_frames, ids_labels)

    ids = models.id_model(util.image_shape)
    ids.summary()
    ids.fit(ids_frames, ids_labels, validation_split=0.2, epochs=5)
