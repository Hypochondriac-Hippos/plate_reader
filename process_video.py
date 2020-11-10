#!/usr/bin/env python3

"""
Process labelled videos into directories of images
"""

import argparse
import json
import os
import random
import string

import cv2

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


def imwrite(filename, img, *args, **kwargs):
    """Wrapper around cv2.imwrite that throws IOError if the write fails."""
    success = cv2.imwrite(filename, img, *args, **kwargs)
    if not success:
        raise IOError("Couldn't write {}".format(filename))

    return success


def progress_bar(n, maximum, width):
    prog = round(n / maximum * width)
    return "#" * prog + "-" * (width - prog)


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

    for file in sorted(os.listdir(VIDEO_DIR)):
        label_file = os.path.join(VIDEO_DIR, file + ".json")
        if os.path.exists(label_file):
            with open(label_file) as f:
                labels = json.load(f)
            labels = intify_keys(labels)
            with video.VideoCapture(os.path.join(VIDEO_DIR, file)) as v:
                for i, frame in enumerate(v):
                    print(
                        f"\r{file} [{progress_bar(i, len(v), 20)}] {i}/{len(v)}", end=""
                    )
                    if i in labels["frames"]:
                        label = labels["frames"][i]
                        imwrite(
                            os.path.join(
                                OUT_DIR,
                                "ids",
                                random.choices(
                                    ("train", "test"),
                                    weights=(TRAIN_TEST_SPLIT, 1 - TRAIN_TEST_SPLIT),
                                )[0],
                                str(label),
                                FILE_NAME_FORMAT.format(
                                    label=label, video=file, frame=i
                                ),
                            ),
                            frame,
                        )
                v.release()
                print()
