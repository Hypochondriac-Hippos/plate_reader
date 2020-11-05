#!/usr/bin/env python

"""
One-hot encoding for various data.
"""

import numpy as np


def id(i):
    """
    One-hot encode a plate ID: 0 => no plate, n => plate N.
    """
    hot = np.zeros((9,))
    hot[i] = 1
    return hot


def un_id(v):
    """
    One-hot decode a plate ID.
    """
    return np.argwhere(v)[0]
