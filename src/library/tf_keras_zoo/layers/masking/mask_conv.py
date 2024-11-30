import tensorflow as tf

from .utils import ComputeOutputMaskMixin1D, apply_mask


class MaskConv1D(ComputeOutputMaskMixin1D, tf.keras.layers.Conv1D):

    def call(self, inputs, mask=None):
        inputs = apply_mask(inputs, mask=mask)
        return super().call(inputs)
