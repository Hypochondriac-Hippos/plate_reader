#!/usr/bin/env python

"""
Neural networks for plate identification (which plate is in frame?) and plate reading
(what is the number on the license plate in frame?).
"""

from keras import models, layers, optimizers


class PlateID:
    def __init__(self, input_shape):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(8, 3, activation="relu", input_shape=input_shape))
        self.model.add(layers.MaxPooling2D())
        self.model.add(layers.Conv2D(8, 3, activation="relu"))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(81, activation="relu"))
        self.model.add(layers.Dense(27, activation="relu"))
        self.model.add(layers.Dense(9, activation="softmax"))
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizers.RMSprop(),
            metrics=["acc"],
        )


class PlateReader:
    pass
