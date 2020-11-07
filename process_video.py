#!/usr/bin/env python3

"""
Process labelled videos into directories of images
"""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import random
import string
import time

import cv2
import numpy as np

import video

VIDEO_DIR = os.path.expanduser("~/Videos/353_recordings")
OUT_DIR = os.path.join(VIDEO_DIR, "images")
TRAIN_TEST_SPLIT = 0.8  # Percentage to put in training dataset
FILE_NAME_FORMAT = "{label}_{video}_{frame}.png"
ID_CLASSES = ("0", "1", "2", "3", "4", "5", "6", "7", "8")


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
    Process a sequence of frames and a labels dict into an iterator of (frame, label, frame number)
    """
    for i, frame in enumerate(video):
        if i in labels["frames"]:
            yield frame, labels["frames"][i], i


def label_plates(video, labels):
    """
    Process a VideoCapture object and labels dict into an ndarray of frames and plate
    number labels.
    """
    pass


def load_data(directory):
    """
    Read through a directory and open the labelled videos it contains.

    Yields a generator (in no particular order) of videos, labels, and filenames
    """
    for root, directories, files in os.walk(directory):
        for file in files:
            video_file = os.path.join(root, file)
            label_file = os.path.join(root, file + ".json")
            if os.path.exists(os.path.join(root, file + ".json")):
                with open(label_file) as f:
                    v = video.VideoCapture(video_file)
                    if v.isOpened():
                        yield v, intify_keys(json.load(f)), label_file


def ensure_output_dirs():
    """Set up the output directory structure, if required."""
    for problem, classes in zip(
        ("ids", "letter_1", "letter_2", "number_1", "number_2"),
        (
            ID_CLASSES,
            string.ascii_uppercase,
            string.ascii_uppercase,
            string.digits,
            string.digits,
        ),
    ):
        for t in ("train", "test"):
            for c in classes:
                os.makedirs(os.path.join(OUT_DIR, problem, t, c), exist_ok=True)


def dump_frames(problem, data, source_video):
    """Given an ndarray and the categorical labels, dump into appropriate output directory."""
    for f, l, n in data:
        imwrite(
            os.path.join(
                OUT_DIR,
                problem,
                random.choices(
                    ("train", "test"), weights=(TRAIN_TEST_SPLIT, 1 - TRAIN_TEST_SPLIT)
                )[0],
                str(l),
                FILE_NAME_FORMAT.format(label=l, video=source_video, frame=n),
            ),
            f,
        )


def imwrite(filename, img, *args, **kwargs):
    """Wrapper around cv2.imwrite that throws IOError if the write fails."""
    success = cv2.imwrite(filename, img, *args, **kwargs)
    if not success:
        raise IOError("Couldn't write {}".format(filename))

    return success


def spinner(q, l, prefix):
    spinner = itertools.cycle(r"-\|/")
    while q.empty():
        with l:
            print(f"\r{prefix} {next(spinner)}", end="")
        time.sleep(0.1)
    print(f"\r{prefix}  ")


if __name__ == "__main__":
    random.seed(1337)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--id", action="store_true", help="train the plate identification network"
    )
    parser.add_argument(
        "-r", "--read", action="store_true", help="train the plate reader network"
    )
    args = parser.parse_args()

    ensure_output_dirs()

    labelled_data = load_data(VIDEO_DIR)

    if args.id:
        all_frames = []
        all_labels = []
        for vid, label, file in labelled_data:
            done = mp.SimpleQueue()
            spinner(done, mp.Lock(), os.path.basename(file))
            dump_frames("ids", label_ids(vid, label), os.path.basename(file))
            done.put(True)
            vid.release()

        frames = np.asarray(all_frames).reshape((-1, all_frames[0][0].shape))
        labels = np.asarray(all_labels).reshape((-1, all_labels[0][0].shape))
