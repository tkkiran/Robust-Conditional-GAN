"""
Origin implementation of conv2D.
"""

import numpy as np
import tensorflow as tf

# import common as lib
from common.ops.sn import spectral_normed_weight

_default_weightnorm = False


def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True


_weights_stdev = None


def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev


def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None


def Conv2D(inputs, input_dim, output_dim, filter_size=3, stride=1, name='Conv2D',
           spectral_normed=False, update_collection=None, reuse=False, inputs_norm=False,
           he_init=True, mask_type=None, weightnorm=None, biases=True, gain=1.):
    """
    Args:
      inputs: Tensor of shape (batch size, height, width, num_channels)
      input_dim:
      output_dim:
      filter_size:
      stride:
      name:
      spectral_normed:
      update_collection:
      reuse:
      inputs_norm:
      he_init:
      mask_type: One of None, 'a', 'b'.
      weightnorm:
      biases:
      gain:

    Returns:
      tensor of shape (batch size, height, width, num channels)
    """
    # with tf.name_scope(name) as scope:
    with tf.variable_scope(name):
        if mask_type is not None:
            mask_type, mask_n_channels = mask_type

            mask = np.ones(
                (filter_size, filter_size, input_dim, output_dim),
                dtype='float32'
            )
            center = filter_size // 2

            # Mask out future locations
            # filter shape is (height, width, input channels, output channels)
            mask[center + 1:, :, :, :] = 0.
            mask[center, center + 1:, :, :] = 0.

            # Mask out future channels
            for i in range(mask_n_channels):
                for j in range(mask_n_channels):
                    if (mask_type == 'a' and i >= j) or (mask_type == 'b' and i > j):
                        mask[center, center, i::mask_n_channels, j::mask_n_channels] = 0.

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        fan_in = input_dim * filter_size ** 2
        fan_out = output_dim * filter_size ** 2 / (stride ** 2)

        if inputs_norm:
            inv_c = np.sqrt(2.0 / fan_in)
            inputs_ = inputs * inv_c
        else:
            inputs_ = inputs

        if mask_type is not None:  # only approximately correct
            fan_in /= 2.
            fan_out /= 2.

        if he_init:
            filters_stdev = np.sqrt(4. / (fan_in + fan_out))
        else:  # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2. / (fan_in + fan_out))

        if _weights_stdev is not None:
            filter_values = uniform(
                _weights_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )
        else:
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )

        # print "WARNING IGNORING GAIN"
        filter_values *= gain

        filters = tf.get_variable(name='Filters',
                                  dtype=tf.float32,
                                  initializer=filter_values)  # tf.glorot_uniform_initializer()

        if weightnorm is None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0, 1, 2)))
            target_norms = tf.get_variable(name='g',
                                           dtype=tf.float32,
                                           initializer=norm_values)

            with tf.name_scope('weightnorm'):
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0, 1, 2]))
                filters = filters * (target_norms / norms)

        if mask_type is not None:
            with tf.name_scope('filter_mask'):
                filters = filters * mask

        if spectral_normed:
            result = tf.nn.conv2d(
                input=inputs_,
                filter=spectral_normed_weight(filters, update_collection=update_collection),
                strides=[1, stride, stride, 1],
                padding='SAME',
                data_format='NHWC'
            )
        else:
            result = tf.nn.conv2d(
                input=inputs_,
                filter=filters,
                strides=[1, stride, stride, 1],
                padding='SAME',
                data_format='NHWC'
            )

        if biases:
            _biases = tf.get_variable(name='Biases', shape=[output_dim, ], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.))

            result = tf.nn.bias_add(result, _biases, data_format='NHWC')

        return result
