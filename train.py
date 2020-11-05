#!/usr/bin/env python

"""
Script to train the plate reader neural networks.
"""

import argparse

import numpy as np

import onehot


def intify_keys(json_labels):
    """
    Process the raw labels parsed from JSON to convert string keys to ints
    """
    return {
        "plates": {
            int(plate): number for plate, number in json_labels["plates"].items()
        },
        "frames": {
            int(float(frame)): plate for frame, plate in json_labels["frames"].items()
        },
    }


def label_ids(video, labels):
    """
    Process a sequence of frames and a labels dict into an ndarray of interesting frames
    and a parallel ndarray of one-hot plate ID labels.
    """
    num_interesting = len(labels["frames"])
    frames = np.empty((num_interesting, *video.shape))
    labels = np.empty((num_interesting, 9))
    j = 0
    for i, frame in enumerate(video):
        if i in labels["frames"]:
            frames[j] = frame
            labels[j] = onehot.id(i)
            j += 1

    assert j == num_interesting

    return frames, labels


def label_plates(video, labels):
    """
    Process a VideoCapture object and labels dict into an ndarray of frames and plate
    number labels.
    """
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--id", action="store_true", help="train the plate identification network"
    )
    parser.add_argument(
        "-r", "--read", action="store_true", help="train the plate reader network"
    )
    args = parser.parse_args()
