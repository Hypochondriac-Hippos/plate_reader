#!/usr/bin/env python2

"""
Read license plates.
"""

import cv2
import numpy as np


def blue(image, blue_h_min=100, blue_h_max=140, saturation_min=100, true=255):
    """Convert to a binary image showing areas that are blue"""
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    return np.logical_and(blue_h_min < h, h < blue_h_max, s > saturation_min)


def isolate_blue_blocks(image, area_min=10, side_ratio=0.5):
    """Return a sequence of masks on the original area showing significant blocks of blue."""
    contours, _ = cv2.findContours(
        blue(image).astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if min(w, h) / max(w, h) > side_ratio and cv2.contourArea(c) > area_min:
            rects.append((x, y, w, h))

    masks = np.zeros_like()

    return filtered
