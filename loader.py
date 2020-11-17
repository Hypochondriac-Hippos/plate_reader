#!/usr/bin/env python

"""
Load image data
"""

import os
import random

import numpy as np
import tensorflow as tf

import util


def list_all_files(root):
    """Get a flat list of all files in a directory and its subdirectories."""
    files = []
    for path, _, fnames in os.walk(root):
        for f in fnames:
            files.append(os.path.join(path, f))

    return files


def load_dataset(root, classes, sample_percent):
    """Return an ndarray of frames and a matching ndarray of one-hot labels."""
    files = []
    labels = []
    for i, c in enumerate(classes):
        all_files = list_all_files(os.path.join(root, c))
        sample = random.sample(all_files, int(np.ceil(sample_percent * len(all_files))))
        files.extend(sample)
        labels.extend([i] * len(sample))

    num_files = len(files)
    frames = np.empty(
        (num_files, util.image_shape[0], util.image_shape[1], util.image_shape[2]),
        dtype=np.uint8,
    )
    labels = np.asarray(tf.one_hot(labels, len(classes)))

    for i, file in enumerate(files):
        frames[i] = util.imread(file)

    return frames, labels
