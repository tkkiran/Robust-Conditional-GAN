"""PGGAN based on ResNet."""

# import numpy as np
import tensorflow as tf
import functools

import os
import sys

sys.path.append(os.getcwd())

import common as lib
import common.ops.conv2d
import common.ops.linear
import common.ops.normalization

# import common.ops.embedding


NORMALIZATION_G = True
NORMALIZATION_D = True


def nonlinearity(x, activation_fn='relu', leakiness=0.2):
    if activation_fn == 'relu':
        return tf.nn.relu(x)
    if activation_fn == 'lrelu':
        assert 0 < leakiness <= 1, "leakiness must be <= 1"
        return tf.maximum(x, leakiness * x)


def Normalize(name, inputs, labels=None, spectral_normed=True):
    with tf.variable_scope(name):
        if ('D.' in name) and NORMALIZATION_D:
            if spectral_normed:
                return inputs
            else:
                # return lib.ops.layernorm.Layernorm(name, [1, 2, 3], inputs)
                return lib.ops.normalization.batch_norm(inputs, fused=True)
        elif ('G.' in name) and NORMALIZATION_G:
            if labels is not None:
                # print('Cond_Batchnorm')
                outputs = lib.ops.normalization.cond_batchnorm(name, [0, 1, 2], inputs, labels=labels, n_labels=10)
                return outputs
            else:
                # print('Batchnorm')
                outputs = lib.ops.normalization.batch_norm(inputs, fused=True)
                return outputs
        else:
            return inputs


def ConvMeanPool(inputs, output_dim, filter_size=3, stride=1, name=None,
                 spectral_normed=False, update_collection=None, inputs_norm=False,
                 he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(inputs, inputs.shape.as_list()[-1], output_dim, filter_size, stride, name,
                                   spectral_normed=spectral_normed,
                                   update_collection=update_collection,
                                   inputs_norm=inputs_norm,
                                   he_init=he_init, biases=biases)
    # output = tf.nn.avg_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    output = tf.add_n(
        [output[:, ::2, ::2, :], output[:, 1::2, ::2, :], output[:, ::2, 1::2, :], output[:, 1::2, 1::2, :]]) / 4.
    return output


def MeanPoolConv(inputs, output_dim, filter_size=3, stride=1, name=None,
                 spectral_normed=False, update_collection=None, inputs_norm=False,
                 he_init=True, biases=True):
    output = inputs
    output = tf.add_n(
        [output[:, ::2, ::2, :], output[:, 1::2, ::2, :], output[:, ::2, 1::2, :], output[:, 1::2, 1::2, :]]) / 4.
    # output = tf.nn.avg_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], output_dim, filter_size, stride, name,
                                   spectral_normed=spectral_normed,
                                   update_collection=update_collection,
                                   inputs_norm=inputs_norm,
                                   he_init=he_init, biases=biases)

    return output


def UpsampleConv(inputs, output_dim, filter_size=3, stride=1, name=None,
                 spectral_normed=False, update_collection=None, inputs_norm=False,
                 he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=3)
    output = tf.depth_to_space(output, 2)
    # w, h = inputs.shape.as_list()[1], inputs.shape.as_list()[2]
    # output = tf.image.resize_images(inputs, [w * 2, h * 2])
    output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], output_dim, filter_size, stride, name,
                                   spectral_normed=spectral_normed,
                                   update_collection=update_collection,
                                   inputs_norm=inputs_norm,
                                   he_init=he_init, biases=biases)

    return output


def ResidualBlock(inputs, input_dim, output_dim, filter_size, name,
                  spectral_normed=False, update_collection=None, inputs_norm=False,
                  resample=None, labels=None, biases=True, activation_fn='relu'):
    """resample: None, 'down', or 'up'.
    """
    if resample == 'down':
        conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2 = functools.partial(ConvMeanPool, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample == 'up':
        conv_1 = functools.partial(UpsampleConv, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample is None:
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim)
        conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample is None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(inputs=inputs, output_dim=output_dim, filter_size=1, name=name + '.Shortcut',
                                 spectral_normed=spectral_normed,
                                 update_collection=update_collection,
                                 inputs_norm=inputs_norm,
                                 he_init=False, biases=biases)

    output = inputs
    output = Normalize(name + '.N1', output, labels=labels, spectral_normed=spectral_normed)
    output = nonlinearity(output, activation_fn=activation_fn)
    # if resample == 'up':
    #     output = nonlinearity(output, activation_fn='relu')
    # else:
    #     output = lrelu(output, leakiness=0.2)

    output = conv_1(inputs=output, filter_size=filter_size, name=name + '.Conv1',
                    spectral_normed=spectral_normed,
                    update_collection=update_collection,
                    inputs_norm=inputs_norm,
                    he_init=True, biases=biases)

    output = Normalize(name + '.N2', output, labels=labels, spectral_normed=spectral_normed)
    output = nonlinearity(output, activation_fn=activation_fn)
    # if resample == 'up':
    #     output = nonlinearity(output, activation_fn='relu')
    # else:
    #     output = lrelu(output, leakiness=0.2)

    output = conv_2(inputs=output, filter_size=filter_size, name=name + '.Conv2',
                    spectral_normed=spectral_normed,
                    update_collection=update_collection,
                    inputs_norm=inputs_norm,
                    he_init=True, biases=biases)

    return shortcut + output


def OptimizedResBlockDisc1(inputs, DIM_D=128, activation_fn='relu',
                           spectral_normed=False, update_collection=None, inputs_norm=False,
                           biases=True):
    conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=inputs.shape.as_list()[-1], output_dim=DIM_D)
    conv_2 = functools.partial(ConvMeanPool, output_dim=DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut(inputs=inputs, output_dim=DIM_D, filter_size=1, name='D.DownBlock.1.Shortcut',
                             spectral_normed=spectral_normed,
                             update_collection=update_collection,
                             inputs_norm=inputs_norm,
                             he_init=False, biases=biases)

    output = inputs
    output = conv_1(inputs=output, filter_size=3, name='D.DownBlock.1.Conv1',
                    spectral_normed=spectral_normed,
                    update_collection=update_collection,
                    inputs_norm=inputs_norm,
                    he_init=True, biases=biases)
    output = nonlinearity(output, activation_fn=activation_fn)
    # output = lrelu(output, leakiness=0.2)
    output = conv_2(inputs=output, filter_size=3, name='D.DownBlock.1.Conv2',
                    spectral_normed=spectral_normed,
                    update_collection=update_collection,
                    inputs_norm=inputs_norm,
                    he_init=True, biases=biases)
    return shortcut + output


# ######## ######## PGGAN ######## ######## #
def get_dim(stage):
    return min(2048 / (2 ** stage), 512)


def Generator_PGGAN(noise, bc, trans=False, alpha=0.01, inputs_norm=False, labels=None, training=True):
    """
    Args:
      noise:
      bc: Count of ResidualBlock.
      trans:
      alpha:
      inputs_norm:
      labels:
      training:
    Return:
    """
    # bc_ = bc

    # (N, 4, 4, 1024)
    output = lib.ops.linear.Linear(noise, noise.shape.as_list()[-1], 4 * 4 * 1024, 'G.Input',
                                   inputs_norm=inputs_norm, biases=True, initialization=None)
    output = tf.reshape(output, [-1, 4, 4, 1024])

    # output = lib.ops.batchnorm.Batchnorm(output)
    output = Normalize('G.N0', output, labels=labels, spectral_normed=True)
    output = nonlinearity(output, activation_fn='relu')
    print('G.Input: {}'.format(output.shape.as_list()))
    # (N, 4, 4, 1024)
    output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], 1024, 3, 1, 'G.Conv',
                                   he_init=True, biases=True)

    # # (N, 8, 8, 1024)
    # output = ResidualBlock(output, 1024, 1024, 3, 'G.UpBlock.1', inputs_norm=inputs_norm, resample='up', labels=labels)
    # print('G.UpBlock.1: {}'.format(output.shape.as_list()))

    for i in range(bc - 1):
        output = ResidualBlock(output, output.shape.as_list()[-1], get_dim(i), 3,
                               'G.UpBlock.{}'.format(i + 1), inputs_norm=inputs_norm, resample='up', labels=labels)
        print('G.UpBlock.{}: {}'.format(i + 1, output.shape.as_list()))

    if trans:
        toRGB1 = ResidualBlock(output, output.shape.as_list()[-1], get_dim(bc - 1), 3,
                               'G.UpBlock.{}'.format(bc), inputs_norm=inputs_norm, resample='up', labels=labels)
        print('G.UpBlock.{}: {}'.format(bc, toRGB1.shape.as_list()))
        toRGB1 = ResidualBlock(toRGB1, toRGB1.shape.as_list()[-1], get_dim(bc - 1), 3,
                               'G.{}_toRGB1'.format(bc), inputs_norm=inputs_norm, resample=None, labels=labels)

        toRGB2 = \
            tf.image.resize_nearest_neighbor(output, [toRGB1.shape.as_list()[1], toRGB1.shape.as_list()[2]])
        toRGB2 = ResidualBlock(toRGB2, toRGB2.shape.as_list()[-1], get_dim(bc - 1), 3,
                               'G.{}_toRGB2'.format(bc), inputs_norm=inputs_norm, resample=None, labels=labels)

        # fade in
        toRGB = (1.0 - alpha) * toRGB2 + alpha * toRGB1
        toRGB = tf.reshape(toRGB, toRGB2.shape.as_list())
        print('G.{}_toRGB: {}'.format(bc, toRGB.shape.as_list()))
    else:
        if bc > 0:
            toRGB = ResidualBlock(output, output.shape.as_list()[-1], get_dim(bc - 1), 3,
                                  'G.UpBlock.{}'.format(bc), inputs_norm=inputs_norm, resample='up', labels=labels)
            print('G.UpBlock.{}: {}'.format(bc, toRGB.shape.as_list()))
        else:
            toRGB = output
        toRGB = ResidualBlock(toRGB, toRGB.shape.as_list()[-1], get_dim(bc - 1), 3,
                              'G.{}_toRGB'.format(bc), inputs_norm=inputs_norm, resample=None, labels=labels)

    # output = lib.ops.batchnorm.Batchnorm(toRGB)
    output = Normalize('G.Output_Normalize', toRGB, labels=labels, spectral_normed=True)
    output = nonlinearity(output, activation_fn='relu')
    output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], 3, 3, 1, 'G.Output', he_init=False)
    # output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], 3, 1, 1, 'G.Output')
    print('G.Output: {}'.format(output.shape.as_list()))

    output = tf.nn.tanh(output)

    return output


def Discriminator_PGGAN(x_var, c_var, bc, trans=False, alpha=0.01, inputs_norm=False, labels=None,
                        update_collection=None, reuse=False):
    """
    Args:
      x_var:
      c_var:
      bc:
      trans:
      alpha:
      inputs_norm:
      labels:
      reuse:
      update_collection:
    Return:
    """
    # imsize = 4 * pow(2, bc)

    if trans:
        fromRGB1 = ResidualBlock(x_var, 3, get_dim(bc - 1), 3, 'D.{}_fromRGB1'.format(bc),
                                 spectral_normed=True,
                                 update_collection=update_collection,
                                 inputs_norm=inputs_norm,
                                 resample=None, biases=True)
        fromRGB1 = ResidualBlock(fromRGB1, get_dim(bc - 1), get_dim(bc - 1), 3, 'D.DownBlock.{}'.format(bc),
                                 spectral_normed=True,
                                 update_collection=update_collection,
                                 inputs_norm=inputs_norm,
                                 resample='down', biases=True)
        print('D.DownBlock.{}: {}'.format(bc, fromRGB1.shape.as_list()))

        fromRGB2 = \
            tf.image.resize_nearest_neighbor(x_var, [fromRGB1.shape.as_list()[1], fromRGB1.shape.as_list()[2]])
        fromRGB2 = ResidualBlock(fromRGB2, 3, get_dim(bc - 1), 3, 'D.{}_fromRGB2'.format(bc),
                                 spectral_normed=True,
                                 update_collection=update_collection,
                                 inputs_norm=inputs_norm,
                                 resample=None, biases=True)
        print('D.{}_fromRGB2: {}'.format(bc, fromRGB2.shape.as_list()))

        x_code = (1.0 - alpha) * fromRGB2 + alpha * fromRGB1
        x_code = tf.reshape(x_code, fromRGB2.shape.as_list())
    else:
        x_code = ResidualBlock(x_var, 3, get_dim(bc - 1), 3, 'D.{}_fromRGB'.format(bc),
                               spectral_normed=True,
                               update_collection=update_collection,
                               inputs_norm=inputs_norm,
                               resample=None, biases=True)

        if bc > 0:
            x_code = ResidualBlock(x_code, get_dim(bc - 1), get_dim(bc - 1), 3, 'D.DownBlock.{}'.format(bc),
                                   spectral_normed=True,
                                   update_collection=update_collection,
                                   inputs_norm=inputs_norm,
                                   resample='down', biases=True)
            print('D.DownBlock.{}: {}'.format(bc, x_code.shape.as_list()))

    for i in range(1, bc):
        x_code = ResidualBlock(x_code, x_code.shape.as_list()[-1], get_dim(bc - 1 - i), 3,
                               'D.DownBlock.{}'.format(bc - i),
                               spectral_normed=True,
                               update_collection=update_collection,
                               inputs_norm=inputs_norm,
                               resample='down', biases=True)
        print('D.DownBlock.{}: {}'.format(bc - i, x_code.shape.as_list()))

    output = ResidualBlock(x_code, x_code.shape.as_list()[-1], get_dim(0), 3,
                           'D.NoneBlock',
                           spectral_normed=True,
                           update_collection=update_collection,
                           inputs_norm=inputs_norm,
                           resample=None, biases=True)
    print('D.NoneBlock: {}'.format(output.shape.as_list()))
    output = nonlinearity(output, activation_fn='relu')

    output = tf.reduce_mean(output, axis=[1, 2])
    logits = lib.ops.linear.Linear(output, output.shape.as_list()[-1], 1, 'D.Output',
                                   spectral_normed=True,
                                   update_collection=update_collection,
                                   inputs_norm=inputs_norm,
                                   biases=True, initialization=None)

    output_wgan = tf.reshape(logits, [-1])

    return output_wgan
