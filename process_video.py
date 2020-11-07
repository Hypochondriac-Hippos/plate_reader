#!/usr/bin/env python3

"""
Process labelled videos into directories of images
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
    frames = np.empty((num_interesting, *video.shape, 3), dtype=video[0].dtype)
    frame_labels = np.empty((num_interesting, 9))
    j = 0
    for i, frame in enumerate(video):
        if i in labels["frames"]:
            frames[j] = frame
            frame_labels[j] = onehot.id(labels["frames"][i])
            j += 1

    assert j == num_interesting

    return frames, frame_labels


def label_plates(video, labels):
    """
    Process a VideoCapture object and labels dict into an ndarray of frames and plate
    number labels.
    """
    pass


def load_data(directory):
    """
    Read through a directory and open the labelled videos it contains.

    Returns: a sequence (in no particular order) of pairs of videos and labels
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
                        output.append((v, intify_keys(json.load(f))))

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

    labelled_data = load_data(VIDEO_DIR)

    if args.id:
        all_frames = []
        all_labels = []
        for vid, label in labelled_data:
            frames, labels = label_ids(vid, label)
            all_frames.append(frames)
            all_labels.append(labels)

        frames = np.asarray(all_frames).reshape((-1, all_frames[0][0].shape))
        labels = np.asarray(all_labels).reshape((-1, all_labels[0][0].shape))
