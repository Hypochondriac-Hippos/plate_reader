#!/usr/bin/env python

"""
Train license plate NNs
"""

import argparse
import datetime
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
    parser.add_argument("-i", "--ids", action="store_true", help="Train plate ID")
    parser.add_argument(
        "-n", "--numbers", action="store_true", help="Train plate numbers"
    )
    parser.add_argument(
        "-p", "--percent", type=float, default=1, help="Percentage of dataset to use"
    )
    args = parser.parse_args()

    ensure_output_dirs()

    if args.ids:
        ids_frames, ids_labels = loader.load_dataset(
            os.path.join(IMAGE_DIR, "ids", "train"),
            util.ID_CLASSES,
            args.percent,
            preprocessor=util.greyscale,
        )

        if args.visualize:
            visualize_dataset(ids_frames, ids_labels)

        ids = models.id_model(ids_frames[0].shape, len(util.ID_CLASSES))
        ids.summary()
        history = ids.fit(ids_frames, ids_labels, validation_split=0.2, epochs=5)

        fig, ax = plt.subplots(ncols=2)
        ax[0].plot(history.history["loss"])
        ax[0].plot(history.history["val_loss"])
        ax[0].set_title("model loss")
        ax[0].set_ylabel("loss")
        ax[0].set_xlabel("epoch")
        ax[0].legend(["train loss", "val loss"], loc="upper left")

        ax[1].plot(history.history["acc"])
        ax[1].plot(history.history["val_acc"])
        ax[1].set_title("model accuracy")
        ax[1].set_ylabel("accuracy (%)")
        ax[1].set_xlabel("epoch")
        ax[1].legend(["train accuracy", "val accuracy"], loc="upper left")
        plt.show()

        now = datetime.datetime.utcnow().replace(second=0, microsecond=0)
        ids.save("trained/ids_{}".format(now.isoformat()))

    if args.numbers:
        for problem, classes in zip(util.PLATE_PROBLEMS, util.PLATE_CLASSES):
            frames, labels = loader.load_dataset(
                os.path.join(IMAGE_DIR, problem, "train"), classes, args.percent
            )

            if args.visualize:
                visualize_dataset(frames, labels)

            reader = models.id_model(util.image_shape, len(classes))
            reader.summary()
            history = reader.fit(frames, labels, validation_split=0.2, epochs=5)

            fig, ax = plt.subplots(ncols=2)
            ax[0].plot(history.history["loss"])
            ax[0].plot(history.history["val_loss"])
            ax[0].set_title("model loss")
            ax[0].set_ylabel("loss")
            ax[0].set_xlabel("epoch")
            ax[0].legend(["train loss", "val loss"], loc="upper left")

            ax[1].plot(history.history["acc"])
            ax[1].plot(history.history["val_acc"])
            ax[1].set_title("model accuracy")
            ax[1].set_ylabel("accuracy (%)")
            ax[1].set_xlabel("epoch")
            ax[1].legend(["train accuracy", "val accuracy"], loc="upper left")
            plt.show()

            now = datetime.datetime.utcnow().replace(second=0, microsecond=0)
            reader.save("trained/{}_{}".format(problem, now.isoformat()))
