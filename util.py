#!/usr/bin/env python

"""
Miscellaneous utilities and data cutting across several files
"""

import os
import sys

import cv2

image_shape = (720, 1280, 3)


def imread(filename, flags=cv2.IMREAD_COLOR):
    """Wrapper around cv2.imread that throws IOError if the read fails."""
    image = cv2.imread(filename, flags)
    if image is None:
        raise IOError("Couldn't read {}".format(filename))

    return image


def imwrite(filename, img, *args, **kwargs):
    """Wrapper around cv2.imwrite that throws IOError if the write fails."""
    success = cv2.imwrite(filename, img, *args, **kwargs)
    if not success:
        raise IOError("Couldn't write {}".format(filename))

    return success


def makedirs(name, *args, **kwargs):
    """Shim to transparently backport exist_ok to Python < 3.2"""
    if "exist_ok" in kwargs and python_lt(3, 2):
        if not os.path.exists(name):
            os.makedirs(name, *args, **kwargs)
    else:
        os.makedirs(name, *args, **kwargs)


def python_lt(major, minor=None, micro=None):
    """Returns true if the python version is less than the given major.minor.patch version."""
    if sys.version_info.major < major:
        return True
    elif sys.version_info.major == major:
        if minor is not None:
            if sys.version_info.minor < minor:
                return True
            elif sys.version_info.minor == minor:
                if micro is not None:
                    if sys.version_info.micro < micro:
                        return True
    return False
