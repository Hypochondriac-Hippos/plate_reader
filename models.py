#!/usr/bin/env python

"""
Neural networks for plate identification (which plate is in frame?) and plate reading
(what is the number on the license plate in frame?).
"""

from __future__ import division

from keras import models, layers, optimizers

try:
    from layers.experimental.preprocessing import Rescaling
except ImportError:
    # Shim to allow running under Python 2, where whichever keras added the rescaling layer
    # isn't supported. Code copied directly from
    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/preprocessing/image_preprocessing.py#L295-L340
    from tensorflow.python.ops import math_ops

    class Rescaling(layers.Layer):
        """Multiply inputs by `scale` and adds `offset`.

        For instance:

        1. To rescale an input in the `[0, 255]` range
        to be in the `[0, 1]` range, you would pass `scale=1./255`.

        2. To rescale an input in the `[0, 255]` range to be in the `[-1, 1]` range,
        you would pass `scale=1./127.5, offset=-1`.

        The rescaling is applied both during training and inference.

        Input shape:
            Arbitrary.
        Output shape:
            Same as input.
        Arguments:
            scale: Float, the scale to apply to the inputs.
            offset: Float, the offset to apply to the inputs.
            name: A string, the name of the layer.
        """

        def __init__(self, scale, offset=0.0, name=None, **kwargs):
            self.scale = scale
            self.offset = offset
            super(Rescaling, self).__init__(name=name, **kwargs)

        def call(self, inputs):
            dtype = inputs.dtype
            scale = math_ops.cast(self.scale, dtype)
            offset = math_ops.cast(self.offset, dtype)
            return math_ops.cast(inputs, dtype) * scale + offset

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            config = {
                "scale": self.scale,
                "offset": self.offset,
            }
            base_config = super(Rescaling, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))


def id_model(input_shape):
    """Set up the plate ID NN"""
    model = models.Sequential()
    model.add(Rescaling(1 / 255, input_shape=input_shape))
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
