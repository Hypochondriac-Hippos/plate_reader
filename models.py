#!/usr/bin/env python

"""
Neural networks for plate identification (which plate is in frame?) and plate reading
(what is the number on the license plate in frame?).
"""

from __future__ import division

from keras import models, layers, optimizers


def id_model(input_shape):
    """Set up the plate ID NN"""
    model = models.Sequential()
    model.add(layers.Conv2D(5, 5, activation="relu"))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(8, 3, activation="relu"))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(81, activation="relu"))
    model.add(layers.Dense(27, activation="relu"))
    model.add(layers.Dense(9, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.RMSprop(),
        metrics=["acc"],
    )
    return model


class PlateReader:
    pass
