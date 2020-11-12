#!/usr/bin/env python3

"""
Process labelled videos into directories of images
"""

import argparse
import json
import os
import random
import string
import sys

import cv2

import video

VIDEO_DIR = os.path.expanduser("~/Videos/353_recordings")
OUT_DIR = os.path.join(VIDEO_DIR, "images")
TRAIN_TEST_SPLIT = 0.8  # Percentage to put in training dataset
FILE_NAME_FORMAT = "{label}_{video}_{frame:04}.png"
ID_CLASSES = ("0", "1", "2", "3", "4", "5", "6", "7", "8")
PLATE_PROBLEMS = ("letter_1", "letter_2", "number_1", "number_2")
PLATE_CLASSES = (
    string.ascii_uppercase,
    string.ascii_uppercase,
    string.digits,
    string.digits,
)


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
        ("ids", *PLATE_PROBLEMS),
        (ID_CLASSES, *PLATE_CLASSES),
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


def train_or_test():
    return random.choices(
        ("train", "test"),
        weights=(TRAIN_TEST_SPLIT, 1 - TRAIN_TEST_SPLIT),
    )[0]


def progress_bar(n, maximum, width):
    prog = round(n / maximum * width)
    return "#" * prog + "-" * (width - prog)


if __name__ == "__main__":
    random.seed(1337)

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Video to process")
    args = parser.parse_args()

    ensure_output_dirs()

    label_file = args.file + ".json"
    if not os.path.exists(args.file):
        print(f"{args.file} does not exist.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(label_file):
        print(f"{args.file} does not have labels.", file=sys.stderr)
        sys.exit(1)

    with open(label_file) as f:
        labels = json.load(f)
    labels = intify_keys(labels)
    with video.VideoCapture(args.file) as v:
        for i, frame in enumerate(v):
            print(f"\r{args.file} [{progress_bar(i, len(v), 20)}] {i}/{len(v)}", end="")

            if i in labels["frames"]:
                id_label = labels["frames"][i]
                imwrite(
                    os.path.join(
                        OUT_DIR,
                        "ids",
                        train_or_test(),
                        str(id_label),
                        FILE_NAME_FORMAT.format(
                            label=id_label, video=os.path.basename(args.file), frame=i
                        ),
                    ),
                    frame,
                )

                if id_label in labels["plates"]:
                    plate = labels["plates"][id_label]
                    for char, place in zip(plate, PLATE_PROBLEMS):
                        imwrite(
                            os.path.join(
                                OUT_DIR,
                                place,
                                train_or_test(),
                                char,
                                FILE_NAME_FORMAT.format(
                                    label=char,
                                    video=os.path.basename(args.file),
                                    frame=i,
                                ),
                            ),
                            frame,
                        )

        print()
