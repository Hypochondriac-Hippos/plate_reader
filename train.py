#!/usr/bin/env python

"""
Script to train the plate reader neural networks.
"""

import argparse
import json
import os

import numpy as np

import onehot
import video

VIDEO_DIR = os.path.expanduser("~/Videos/353_recordings")


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


def load_data(directory):
    """
    Read through a directory and open the labelled videos it contains.

    Returns: sequences of opened videos and labels
    """
    output = []
    for root, directories, files in os.walk(directory):
        for file in files:
            video_file = os.path.join(root, file)
            label_file = os.path.join(root, file + ".json")
            if os.path.exists(os.path.join(root, file + ".json")):
                with open(label_file) as f:
                    v = video.VideoCapture(video_file)
                    if v.isOpened():
                        output.append((v, json.load(f)))

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--id", action="store_true", help="train the plate identification network"
    )
    parser.add_argument(
        "-r", "--read", action="store_true", help="train the plate reader network"
    )
    args = parser.parse_args()

    videos, labels = load_data(VIDEO_DIR)
