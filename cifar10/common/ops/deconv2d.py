"""

"""

import numpy as np
import tensorflow as tf

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


def Deconv2D(inputs, in_channels, output_channels, filter_size, stride=2, padding='SAME', he_init=True,
             weight_norm=None, gain=1., mask_type=None, biases=True, name='Deconv2D'):
    """

    Args:
      inputs: Tensor of shape (batch size, height, width, in_channels).
      in_channels:
      output_channels:
      filter_size:
      stride:
      padding:
      he_init:
      weight_norm:
      gain:
      mask_type: One of None, 'a', 'b'.
      biases:
      name:

    Returns:
      tensor of shape (batch_size, 2*height, 2*width, output_channels)
    """
    with tf.variable_scope(name):
        if mask_type is not None:
            raise Exception('Unsupported configuration in Deconv2D!')

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        # stride = 2
        fan_in = in_channels * filter_size ** 2 / (stride ** 2)
        fan_out = output_channels * filter_size ** 2

        if he_init:
            filters_stdev = np.sqrt(4. / (fan_in + fan_out))
        else:  # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2. / (fan_in + fan_out))

        if _weights_stdev is not None:
            filter_values = uniform(
                _weights_stdev,
                (filter_size, filter_size, output_channels, in_channels)
            )
        else:
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, output_channels, in_channels)
            )

        filter_values *= gain

        filters = tf.get_variable(name='Filters',
                                  dtype=tf.float32,
                                  initializer=filter_values)  # tf.glorot_uniform_initializer()

        if weight_norm is None:
            weight_norm = _default_weightnorm
        if weight_norm:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0, 1, 3)))
            target_norms = tf.get_variable(name='g',
                                           dtype=tf.float32,
                                           initializer=norm_values)
            with tf.name_scope('weight_norm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0, 1, 3]))
                filters = filters * tf.expand_dims(target_norms / norms, 1)

        # inputs = tf.transpose(inputs, [0, 2, 3, 1], name='NCHW_to_NHWC')
        input_shape = tf.shape(inputs)
        output_shape = tf.stack([input_shape[0], 2 * input_shape[1], 2 * input_shape[2], output_channels])

        result = tf.nn.conv2d_transpose(
            value=inputs,
            filter=filters,
            output_shape=output_shape,
            strides=[1, stride, stride, 1],
            padding=padding,
            data_format="NHWC"
        )

        if biases:
            _biases = tf.get_variable(name='Biases', shape=[output_channels, ], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.))
            result = tf.nn.bias_add(result, _biases)

        # result = tf.transpose(result, [0, 3, 1, 2], name='NHWC_to_NCHW')

        return result
