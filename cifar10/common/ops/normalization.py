"""

"""

import tensorflow as tf


def batch_norm(inputs, decay=0.9, epsilon=1e-5, is_training=True, fused=True):
    output = \
        tf.contrib.layers.batch_norm(
            inputs,
            decay=decay,
            center=True,
            scale=True,
            epsilon=epsilon,
            updates_collections=None,
            is_training=is_training,
            fused=fused,
            data_format='NHWC',
            zero_debias_moving_mean=True,
            scope='BatchNorm'
        )

    return output


def cond_batchnorm(name, axes, inputs, is_training=None, stats_iter=None, update_moving_stats=True, fused=True,
                   labels=None, n_labels=None):
    """Conditional Batchnorm (dumoulin et al 2016) for BHWC conv filtermaps
    Args:
      name:
      axes:
      inputs: Tensor of shape (batch size, height, width, num_channels)
      is_training:
      stats_iter:
      update_moving_stats:
      fused:
      labels:
      n_labels:

    Returns:
    """
    with tf.variable_scope('CondBatchNorm'):
        if axes != [0, 1, 2]:
            raise Exception('Axes is not supported in Conditional BatchNorm!')

        mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
        shape = mean.get_shape().as_list()  # shape is [1, 1, 1, n]
        offset_m = tf.get_variable(name='offset', shape=[n_labels, shape[3]], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.))
        scale_m = tf.get_variable(name='scale', shape=[n_labels, shape[3]], dtype=tf.float32,
                                  initializer=tf.constant_initializer(1.))

        offset = tf.nn.embedding_lookup(offset_m, labels)
        scale = tf.nn.embedding_lookup(scale_m, labels)

        result = tf.nn.batch_normalization(inputs, mean, var, offset[:, None, None, :], scale[:, None, None, :], 1e-5)

        return result


def layer_norm(name, norm_axes, inputs):
    """
    Args:
      name:
      norm_axes:
      inputs:

    Returns:
    """
    result = \
        tf.contrib.layers.layer_norm(inputs,
                                     center=True,
                                     scale=True,
                                     activation_fn=None,
                                     reuse=None,
                                     variables_collections=None,
                                     outputs_collections=None,
                                     trainable=True,
                                     begin_norm_axis=1,
                                     begin_params_axis=-1,
                                     scope=name)

    # with tf.variable_scope('LayerNorm'):
    #     mean, var = tf.nn.moments(inputs, norm_axes, keep_dims=True)
    #
    #     # Assume the 'neurons' axis is the last of norm_axes.
    #     # This is the case for fully-connected and BHWC conv layers.
    #     n_neurons = inputs.get_shape().as_list()[norm_axes[-1]]
    #
    #     offset = tf.get_variable(name='offset', shape=[n_neurons, ], dtype=tf.float32,
    #                              initializer=tf.constant_initializer(0.))
    #     scale = tf.get_variable(name='scale', shape=[n_neurons, ], dtype=tf.float32,
    #                             initializer=tf.constant_initializer(1.))
    #
    #     # Add broadcasting dims to offset and scale (e.g. BCHW conv data)
    #     offset = tf.reshape(offset, [1 for _ in range(len(norm_axes) - 1)] + [-1])
    #     scale = tf.reshape(scale, [1 for _ in range(len(norm_axes) - 1)] + [-1])
    #
    #     result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)

    return result


def instance_norm(inputs, epsilon=1e-06):
    output = \
        tf.contrib.layers.instance_norm(
            inputs,
            center=True,
            scale=True,
            epsilon=epsilon,
            activation_fn=None,
            param_initializers=None,
            reuse=None,
            variables_collections=None,
            outputs_collections=None,
            trainable=True,
            data_format='NHWC',
            scope=None
        )

    return output


def pixel_norm(inputs, eps=1e-8):
    """From PGGAN.
    Args:
      inputs: (B, H, W, C)
      eps:

    Returns:
    """
    print('Using Pixelnorm...')
    # outputs = inputs / tf.sqrt(tf.reduce_mean(inputs ** 2, axis=3, keepdims=True) + eps)

    alpha = 1.0 / tf.sqrt(tf.reduce_mean(inputs * inputs, axis=3, keepdims=True) + eps)  # (B, H, W, 1)
    alpha = tf.tile(alpha, multiples=[1, 1, 1, inputs.shape.as_list()[3]])
    outputs = alpha * inputs

    return outputs
