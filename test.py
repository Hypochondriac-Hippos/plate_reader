#!/usr/bin/env python

"""
Test trained NNs
"""

import argparse
import os

import tensorflow as tf

import loader
import util

IMAGE_DIR = os.path.expanduser("~/Videos/353_recordings/images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ids", help="path to saved ID NN")
    args = parser.parse_args()

    if args.ids:
        ids_frames, ids_labels = loader.load_dataset(
            os.path.join(IMAGE_DIR, "ids", "test"),
            util.ID_CLASSES,
            0.1,
            preprocessor=util.greyscale,
        )

        model = tf.keras.models.load_model(args.ids)
        results = model.evaluate(ids_frames, ids_labels)
        print(dict(zip(model.metrics_names, results)))
