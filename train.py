#!/usr/bin/env python3

"""
Train license plate NNs
"""

import os

IMAGE_DIR = os.path.expanduser("~/Videos/353_recordings/images")
ID_OUT = os.path.abspath("./trained/id_classifier")
PLATE_OUT = os.path.abspath("./trained/plate_classifier")


def ensure_output_dirs():
    os.makedirs(os.path.dirname(ID_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(PLATE_OUT), exist_ok=True)


if __name__ == "__main__":
    ensure_output_dirs()
