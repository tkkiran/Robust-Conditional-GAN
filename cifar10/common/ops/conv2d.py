"""
Convolution for data in format of 'NWHC'.
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
           conv_type='conv2d', channel_multiplier=0, padding='SAME',
           spectral_normed=False, update_collection=None, inputs_norm=False, he_init=True,
           mask_type=None, weightnorm=None, biases=True, gain=1.):
    """
    Args:
      inputs: Tensor of shape (batch size, height, width, in_channels).
      input_dim: in_channels.
      output_dim:
      filter_size:
      stride: Integer (for [1, stride, stride, 1]) or tuple/list.
      name:
      conv_type: conv2d, depthwise_conv2d, separable_conv2d.
      channel_multiplier:
      padding:
      spectral_normed:
      update_collection:
      inputs_norm: From PGGAN.
      he_init:
      mask_type: One of None, 'a', 'b'.
      weightnorm:
      biases:
      gain:

    Returns:
      tensor of shape (batch_size, out_height, out_width, output_dim)
    """
    # with tf.name_scope(name) as scope:
    with tf.variable_scope(name):
        if conv_type != "conv2d":
            assert (channel_multiplier > 0, 'channel_multiplier should >0!')

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
            filters_stdev = np.sqrt(2. / (fan_in + fan_out))  # tf.glorot_uniform_initializer()

        if _weights_stdev is not None:
            filter_values = uniform(
                _weights_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )

            if channel_multiplier > 0:
                depthwise_filter_values = uniform(
                    _weights_stdev,
                    ([filter_size, filter_size, input_dim, channel_multiplier])
                )
                pointwise_filter_values = uniform(
                    _weights_stdev,
                    ([1, 1, input_dim * channel_multiplier, output_dim])
                )
        else:
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )

            if channel_multiplier > 0:
                depthwise_filter_values = uniform(
                    filters_stdev,
                    (filter_size, filter_size, input_dim, channel_multiplier)
                )
                pointwise_filter_values = uniform(
                    filters_stdev,
                    (1, 1, input_dim * channel_multiplier, output_dim)
                )

        # print "WARNING IGNORING GAIN"
        filter_values *= gain

        filters = tf.get_variable(name='Filters',
                                  dtype=tf.float32,
                                  initializer=filter_values)
        if channel_multiplier > 0:
            depthwise_filters = tf.get_variable(name='depthwise_filters',
                                                dtype=tf.float32,
                                                initializer=depthwise_filter_values)
            pointwise_filters = tf.get_variable(name='pointwise_filters',
                                                dtype=tf.float32,
                                                initializer=pointwise_filter_values)

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
            with tf.variable_scope('filters'):
                filters = spectral_normed_weight(filters, update_collection=update_collection)

            if channel_multiplier > 0:
                with tf.variable_scope('depthwise_filters'):
                    depthwise_filters = spectral_normed_weight(depthwise_filters, update_collection=update_collection)

                with tf.variable_scope('pointwise_filters'):
                    pointwise_filters = spectral_normed_weight(pointwise_filters, update_collection=update_collection)

        if conv_type == 'conv2d':
            result = tf.nn.conv2d(
                input=inputs_,
                filter=filters,
                strides=[1, stride, stride, 1],
                padding=padding,
                data_format='NHWC'
            )
        elif conv_type == 'depthwise_conv2d':
            result = tf.nn.depthwise_conv2d(
                input=inputs_,
                filter=depthwise_filters,
                strides=[1, stride, stride, 1],
                padding=padding,
                rate=None,
                name=None,
                data_format='NHWC'
            )
        elif conv_type == 'separable_conv2d':
            result = tf.nn.separable_conv2d(
                inputs_,
                depthwise_filter=depthwise_filters,
                pointwise_filter=pointwise_filters,
                strides=[1, stride, stride, 1],
                padding=padding,
                rate=None,
                name=None,
                data_format='NHWC'
            )
        else:
            raise NotImplementedError('{0} is not supported!'.format(conv_type))

        if biases:
            _biases = tf.get_variable(name='Biases', shape=[output_dim, ], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.))

            result = tf.nn.bias_add(result, _biases, data_format='NHWC')

        return result
