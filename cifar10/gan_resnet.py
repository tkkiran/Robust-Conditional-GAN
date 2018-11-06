"""
Code adapted from https://github.com/watsonyanghx/GAN_Lib_Tensorflow

SNGAN with projection ResNet for conditional generation of CIFAR-10
"""

from datetime import datetime
import os
import sys
import logging

sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

import time
import functools
import locale

import common.misc

import common.inception.inception_score_
import common as lib
import common.ops.linear
import common.ops.conv2d
import common.ops.embedding
import common.ops.normalization
import common.plot

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the extracted files here!
DATA_DIR = '../data/cifar10/cifar-10-batches-py/'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')


flags = tf.app.flags

flags.DEFINE_string("dataset", 'cifar', "Dataset")
flags.DEFINE_string("algorithm", 'rcgan', "Algorithm [rcgan, rcgan-u, biased, unbiased]")
flags.DEFINE_float("alpha", 0.8, "1 - noise level")
flags.DEFINE_string("run", '0', "run name")
flags.DEFINE_string("log_file", None, "logging file")
flags.DEFINE_string("parent_dir", '.', "parent directory for checkpoints")
flags.DEFINE_string("expt_dir", None, "directory for expts")

flags.DEFINE_integer("inception_freq", 2500, "frequncy of inception score calculation")
flags.DEFINE_integer("sample_freq", 2500, "frequncy of dev cost calc. and sample pics")
flags.DEFINE_integer("generated_label_accuracy_freq", 2500, "frequncy of generated label accruacy")
flags.DEFINE_integer("sample_save_freq", 0, "frequncy of saving samples")

flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("niters", 50000, "no. of batches")
flags.DEFINE_float("lr", 2.0e-4, "learning rate")
flags.DEFINE_integer("ngpus", 2, "no. of gpus")
flags.DEFINE_boolean("multi_gpu_multi_batch", True, 
                     'whether to multiply batch_size with number of gpus'
                     'and divide nof. iterations by nof. gpus')

flags.DEFINE_boolean("confuse_init", False, "whether to initialize confusion matrix with identity")
flags.DEFINE_float("confuse_init_diag", 0.2, "intial confusion matrix with diagonal entry")
flags.DEFINE_float("confuse_multiplier", 1.0, "learning rate multiplier for learnable confusion matrix ")
flags.DEFINE_boolean("confuse_lr_decay", False, 'whether to decay confusion matrix estimation learning rate')

flags.DEFINE_boolean("perm_classifier", False, 'whether to real fake classifier or not.')
flags.DEFINE_float("perm_multiplier", 1.0, 'whether to real fake classifier or not.')
flags.DEFINE_string("perm_type", 'linear', 'type of real fake classifier to use [linear, 2layer].')

flags.DEFINE_boolean("restore", True, 'whether to restore from past checkpoint')

flags.DEFINE_boolean(
    "perm_gen_label_acc", False, 'whether to calculate generated label accuracy'
    'by taking min. value over all permutation of labels')

flags.DEFINE_string("log_level", 'info', 'logging level [info, debug]')


FLAGS = flags.FLAGS

if FLAGS.log_file is None:
    raise ValueError('flag log_file is required')

# dataset = str(sys.argv[1]) 
# ALGORITHM = str(sys.argv[2])
# ALPHA = float(sys.argv[3])
# run = str(sys.argv[4]) 
# log_file = str(sys.argv[5])

dataset = FLAGS.dataset
ALGORITHM = FLAGS.algorithm
ALPHA = FLAGS.alpha
run = FLAGS.run
log_file = FLAGS.log_file

if FLAGS.log_level == 'debug':
    log_level = logging.DEBUG
elif FLAGS.log_level == 'info':
    log_level = logging.INFO

logging.basicConfig(
    filename=log_file, level=log_level,
    format='%(asctime)s %(levelname)-8s %(message)s')

logging.info('alpha = {}'.format(ALPHA))
C_ALPHA = ((1-ALPHA)/9.0)*np.ones((10,10)) + (ALPHA - (1-ALPHA)/9.0)*np.eye((10))

if dataset == "cifar":
    import common.data.cifar10 as dataset_
    OUTPUT_DIM = 3072  # Number of pixels in CIFAR10 (32*32*3)    
    IMG_SIZE = 32
    IMG_DIM = 3
    INCEPTION_FREQUENCY = 5000 # 1000  # How frequently to calculate Inception score
    SAMPLE_FREQUENCY = 100
    SAMPLE_SAVE_FREQUENCY = 0 # 5000
    GENERATED_LABEL_ACCURACY_FREQ = 5000
    DIR = os.path.join(FLAGS.parent_dir, ALGORITHM + '_alpha' + str(ALPHA)+ '_run-' +  run + '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))
if dataset == "mnist":
    import common.data.mnist10 as dataset_
    OUTPUT_DIM = 1024  # Number of pixels in CIFAR10 (32*32*3)    
    IMG_SIZE = 32
    IMG_DIM = 1
    INCEPTION_FREQUENCY = 10000000  # How frequently to calculate Inception score
    SAMPLE_FREQUENCY = 50
    SAMPLE_SAVE_FREQUENCY = 1000
    DIR = './run_mnist_' + ALGORITHM + '_' + str(ALPHA) + '_' +  run    

if FLAGS.expt_dir is not None:
    DIR = '{}/{}'.format(FLAGS.parent_dir, FLAGS.expt_dir)

INCEPTION_FREQUENCY = FLAGS.inception_freq
SAMPLE_FREQUENCY = FLAGS.sample_freq
SAMPLE_SAVE_FREQUENCY = FLAGS.sample_save_freq
GENERATED_LABEL_ACCURACY_FREQ = FLAGS.generated_label_accuracy_freq

if not os.path.exists(DIR):
    os.mkdir(DIR)
DIR = DIR + '/'  
    
BATCH_SIZE = 64  # Critic batch size
GEN_BS_MULTIPLE = 2  # Generator batch size, as a multiple of BATCH_SIZE

ITERS = 100000  # How many iterations to train for
ITERS = 50000  # How many iterations to train for
# ITERS = 3000  # DEBUG
# ITERS = 20  # DEBUG

BATCH_SIZE = FLAGS.batch_size
ITERS = FLAGS.niters

Z_DIM = 128 # dimension of the noise input to generator 
DIM_G = 128  # Generator dimensionality
DIM_D = 128  # Critic dimensionality
NORMALIZATION_G = True  # Use batchnorm in generator?
NORMALIZATION_D = False  # Use batchnorm (or layernorm) in critic?
LR = 0.0002  # 2e-4  # Initial learning rate
DECAY = True  # Whether to decay LR over learning
N_CRITIC = 5  # 5  # Critic steps per generator steps

LR = FLAGS.lr

CONDITIONAL = True  # Whether to train a conditional or unconditional model
ACGAN = False  # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1.  # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1  # How to scale generator's ACGAN loss relative to WGAN loss

# SPECTRAL_NORM_UPDATE_OPS = "spectral_norm_update_ops"
# WORD2VEC_FILE = np.load(os.path.join(DATA_DIR, 'glove_y.npy')).astype('float32')
WORD2VEC_FILE = None
VOCAB_SIZE = 10
EMBEDDING_DIM = 300  # 620
CHECKPOINT_DIR = os.path.join(DIR, 'checkpoint')
LOSS_TYPE = 'HINGE'  # 'Goodfellow', 'HINGE', 'WGAN', 'WGAN-GP'
SOFT_PLUS = False
RESTORE = True
CONCAT_LABEL = False  # whether concat label to 'z' in Generator.

RESTORE = FLAGS.restore

if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    logging.warning("WARNING! Conditional model without normalization in D might be effectively unconditional!")

N_GPUS = FLAGS.ngpus
if N_GPUS not in [1, 2]:
    raise Exception('Only 1 or 2 GPUs supported!')
DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]
if len(DEVICES) == 1:  # Hack because the code assumes 2 GPUs
    DEVICES = [DEVICES[0], DEVICES[0]]

if FLAGS.multi_gpu_multi_batch:
    BATCH_SIZE = BATCH_SIZE*N_GPUS
    ITERS = ITERS//N_GPUS

lib.print_model_settings(locals().copy())

common.misc.record_setting(os.path.join(DIR, 'scripts'))


def nonlinearity(x, activation_fn='relu', leakiness=0.2):
    if activation_fn == 'relu':
        return tf.nn.relu(x)
    if activation_fn == 'lrelu':
        assert 0 < leakiness <= 1, "leakiness must be <= 1"
        return tf.maximum(x, leakiness * x)


def Normalize(name, inputs, labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm,
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""

    with tf.variable_scope(name):
        if not CONDITIONAL:
            labels = None
        if CONDITIONAL and ACGAN and ('D.' in name):
            labels = None

        if ('D.' in name) and NORMALIZATION_D:
            return lib.ops.normalization.layer_norm(name, [1, 2, 3], inputs)
        elif ('G.' in name) and NORMALIZATION_G:
            if labels is not None:
                outputs = lib.ops.normalization.cond_batchnorm(name, [0, 1, 2], inputs, labels=labels, n_labels=10)
                return outputs
            else:
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
                                   he_init=he_init, biases=biases)

    return output


def ResidualBlock(inputs, input_dim, output_dim, filter_size, name,
                  spectral_normed=False, update_collection=None, inputs_norm=False,
                  resample=None, labels=None, biases=True):
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
                                 he_init=False, biases=biases)

    output = inputs
    output = Normalize(name + '.N1', output, labels=labels)
    output = nonlinearity(output)
    # if resample == 'up':
    #     output = nonlinearity(output)
    # else:
    #     output = lrelu(output, leakiness=0.2)

    output = conv_1(inputs=output, filter_size=filter_size, name=name + '.Conv1',
                    spectral_normed=spectral_normed,
                    update_collection=update_collection,
                    he_init=True, biases=biases)

    output = Normalize(name + '.N2', output, labels=labels)
    output = nonlinearity(output)
    # if resample == 'up':
    #     output = nonlinearity(output)
    # else:
    #     output = lrelu(output, leakiness=0.2)

    output = conv_2(inputs=output, filter_size=filter_size, name=name + '.Conv2',
                    spectral_normed=spectral_normed,
                    update_collection=update_collection,
                    he_init=True, biases=biases)

    return shortcut + output


def OptimizedResBlockDisc1(inputs,
                           spectral_normed=False, update_collection=None, inputs_norm=False,
                           biases=True):
    conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=IMG_DIM, output_dim=DIM_D)
    conv_2 = functools.partial(ConvMeanPool, output_dim=DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut(inputs=inputs, output_dim=DIM_D, filter_size=1, name='D.Block.1.Shortcut',
                             spectral_normed=spectral_normed,
                             update_collection=update_collection,
                             he_init=False, biases=biases)

    output = inputs
    output = conv_1(inputs=output, filter_size=3, name='D.Block.1.Conv1',
                    spectral_normed=spectral_normed,
                    update_collection=update_collection,
                    he_init=True, biases=biases)
    output = nonlinearity(output)
    # output = lrelu(output, leakiness=0.2)
    output = conv_2(inputs=output, filter_size=3, name='D.Block.1.Conv2',
                    spectral_normed=spectral_normed,
                    update_collection=update_collection,
                    he_init=True, biases=biases)
    return shortcut + output


def Generator(n_samples_, labels, noise=None, reuse=False):
    with tf.variable_scope("Generator", reuse=reuse):
        if noise is None:
            noise = tf.random_normal([n_samples_, Z_DIM])
        output = lib.ops.linear.Linear(noise, 128, 4 * 4 * DIM_G * 8, 'G.Input')
        output = tf.reshape(output, [-1, 4, 4, DIM_G * 8])
        output = ResidualBlock(output, DIM_G * 8, DIM_G * 2, 3, 'G.Block.1', resample='up', labels=labels, biases=True)
        output = ResidualBlock(output, DIM_G * 2, DIM_G * 2, 3, 'G.Block.2', resample='up', labels=labels, biases=True)
        output = ResidualBlock(output, DIM_G * 2, DIM_G * 2, 3, 'G.Block.3', resample='up', labels=labels, biases=True)
        output = Normalize('G.OutputNorm', output, labels)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(output, DIM_G * 2, IMG_DIM, 3, 1, 'G.Output', he_init=False)
        output = tf.tanh(output)
        # return tf.reshape(tf.transpose(output, [0, 3, 1, 2], name='NHWC_to_NCHW'), [-1, OUTPUT_DIM])
        return tf.reshape(output, [-1, OUTPUT_DIM])


def Discriminator(inputs, labels, update_collection=None, reuse=False):
    with tf.variable_scope("Discriminator", reuse=reuse):
        if ALGORITHM in ("unbiased", "rcgan-u"):
            labels_disc = None
        else:
            labels_disc = labels
        output = tf.reshape(inputs, [-1, IMG_SIZE, IMG_SIZE, IMG_DIM])
        output = OptimizedResBlockDisc1(output,
                                        spectral_normed=True,
                                        update_collection=update_collection,
                                        biases=True)
        output = ResidualBlock(output, DIM_D, DIM_D, 3, 'D.Block.2',
                               spectral_normed=True,
                               update_collection=update_collection,
                               resample='down', labels=labels_disc, biases=True)
        output = ResidualBlock(output, DIM_D, DIM_D, 3, 'D.Block.3',
                               spectral_normed=True,
                               update_collection=update_collection,
                               resample=None, labels=labels_disc, biases=True)
        output = ResidualBlock(output, DIM_D, DIM_D, 3, 'D.Block.4',
                               spectral_normed=True,
                               update_collection=update_collection,
                               resample=None, labels=labels_disc, biases=True)
        output = ResidualBlock(output, DIM_D, DIM_D, 3, 'D.Block.5',
                               spectral_normed=True,
                               update_collection=update_collection,
                               resample=None, labels=labels_disc, biases=True)
        output = ResidualBlock(output, DIM_D, DIM_D, 3, 'D.Block.6',
                               spectral_normed=True,
                               update_collection=update_collection,
                               resample=None, labels=labels_disc, biases=True)
        output = nonlinearity(output)
        # output = lrelu(output, leakiness=0.2)
        output = tf.reduce_mean(output, axis=[1, 2])
        output_wgan = lib.ops.linear.Linear(output, DIM_D, 1, 'D.Output',
                                            spectral_normed=True,
                                            update_collection=update_collection)
        output_wgan = tf.reshape(output_wgan, [-1])
        return output, output_wgan
            
def Discriminator_projection(labels, update_collection=None, reuse=False):
    with tf.variable_scope("Discriminator", reuse=reuse):
        embedding_y = lib.ops.embedding.embed_y(labels, VOCAB_SIZE, EMBEDDING_DIM, word2vec_file=WORD2VEC_FILE)
        embedding_y = lib.ops.linear.Linear(embedding_y, EMBEDDING_DIM, DIM_D, 'D.Embedding_y',
                                            spectral_normed=True,
                                            update_collection=update_collection,
                                            biases=True)  # (N, DIM_D)
        return embedding_y   


def generated_label_accuracy(samples, labels, confusion_matrix=None):
    with tf.gfile.GFile('../wenxinxu_resnet-in-tensorflow/resnet-110/graph_optimized.pb', 'rb') as f:
      graph_def_optimized = tf.GraphDef()
      graph_def_optimized.ParseFromString(f.read())

    if confusion_matrix is not None:
        _confusion_matrix = confusion_matrix
        confusion_matrix = np.zeros_like(confusion_matrix, dtype=int)
        confusion_matrix[
            np.arange(confusion_matrix.shape[0]),
            np.argmax(_confusion_matrix, axis=-1)] = 1
        _labels = labels
        labels = np.zeros([_labels.shape[0], VOCAB_SIZE], dtype=float)
        labels[np.arange(labels.shape[0]), _labels] = 1
        labels[:] = labels.dot(confusion_matrix)
        labels = np.argmax(labels, axis=-1)
        
    G = tf.Graph()
    with G.as_default():
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
      config = tf.ConfigProto(device_count = {'GPU': 1}, gpu_options=gpu_options)
      num_test = 100
      with tf.Session(config=config) as sess:
        pred_softmax = tf.import_graph_def(graph_def_optimized, return_elements=['infer_softmax:0'])
        x = G.get_tensor_by_name('import/resnet_test_batch:0')

        softmax = sess.run(pred_softmax, feed_dict={x: samples})
        acc = (labels == np.argmax(softmax, axis=-1)).astype(float).mean()

        logging.info('generated label accuracy: {}'.format(acc))

        return acc


def perm_classifier(x, reuse=False):
    if FLAGS.perm_type == 'linear':
        with tf.variable_scope("Discriminator", reuse=reuse):
            # 1 layer NN
            hidden_layer = lib.ops.linear.Linear(
                tf.reshape(x, [-1, OUTPUT_DIM]), 
                OUTPUT_DIM, VOCAB_SIZE, 'D.d_perm_classifier_h1',
                spectral_normed=True, biases=True, reuse=reuse)
            logits = hidden_layer
    elif FLAGS.perm_type == '2layer':
        with tf.variable_scope("Discriminator", reuse=reuse):
            # 1 layer NN
            hidden_layer = lib.ops.linear.Linear(
                tf.reshape(x, [-1, OUTPUT_DIM]), 
                OUTPUT_DIM, 128, 'D.d_perm_classifier_h1',
                spectral_normed=True, biases=True, reuse=reuse)
            hidden_layer = lib.ops.linear.Linear(
                hidden_layer, 
                128, VOCAB_SIZE, 'D.d_perm_classifier_h2',
                spectral_normed=True, biases=True, reuse=reuse)

            logits = hidden_layer
    else:
        raise ValueError('Unknown perm_type {}'.format(FLAGS.perm_type))

    return logits


def sigmoid_cross_entropy_with_logits(x, y):
  try:
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
  except:
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)


def main(_):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:

        # confusion matrix variable for estimation
        if ALGORITHM == 'rcgan-u':
          if not FLAGS.confuse_init:
            confusion_logits = tf.get_variable(
                'confusion_logits', dtype=tf.float32, shape=[VOCAB_SIZE, VOCAB_SIZE],
                trainable=True)
          else:
            if FLAGS.confuse_init_diag > 0.99 and VOCAB_SIZE == 10.:
                aa = 7.0
            else:
                aa = np.log(VOCAB_SIZE*FLAGS.confuse_init_diag/
                            (1.-FLAGS.confuse_init_diag))
            aa = min(7.0, aa)
            
            mean = 0.0 # 0.2/VOCAB_SIZE
            confuse_init = (0 - aa/VOCAB_SIZE + mean)*np.ones(
                [VOCAB_SIZE, VOCAB_SIZE], dtype=np.float32)
            np.fill_diagonal(confuse_init, (aa - (aa/VOCAB_SIZE) + mean))

            confusion_logits = tf.get_variable(
                'confusion_logits', dtype=tf.float32, 
                initializer=tf.constant_initializer(confuse_init),
                shape=[VOCAB_SIZE, VOCAB_SIZE], trainable=True)

          confusion_matrix = tf.nn.softmax(confusion_logits, dim=-1)
        else:
          confusion_matrix = tf.constant(C_ALPHA.astype(np.float32))

        _iteration = tf.placeholder(tf.int32, shape=None)
        all_real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
        all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        labels_splits = tf.split(all_real_labels, len(DEVICES), axis=0)

        all_random_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='d1')
        labels_random_splits = tf.split(all_random_labels, len(DEVICES), axis=0)

        all_labels_biased = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='d2')
        labels_biased_splits = tf.split(all_labels_biased, len(DEVICES), axis=0)

        all_labels_inv_weights = tf.placeholder(tf.float32, shape=[BATCH_SIZE,VOCAB_SIZE])
        labels_inv_weights_splits = tf.split(all_labels_inv_weights, len(DEVICES), axis=0)
            
        fake_data_splits = []
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                if i > 0:
                    fake_data_splits.append(Generator(int(BATCH_SIZE / len(DEVICES)), labels_random_splits[i], reuse=True))
                else:
                    fake_data_splits.append(Generator(int(BATCH_SIZE / len(DEVICES)), labels_random_splits[i]))

        all_real_data = tf.reshape(2 * ((tf.cast(all_real_data_int, tf.float32) / 256.) - .5), [BATCH_SIZE, OUTPUT_DIM])
        all_real_data += tf.random_uniform(shape=[BATCH_SIZE, OUTPUT_DIM], minval=0., maxval=1. / 128)  # dequantize
        all_real_data = tf.reshape(
            tf.transpose(tf.reshape(all_real_data, [-1, IMG_DIM, IMG_SIZE, IMG_SIZE]), perm=[0, 2, 3, 1]), [-1, OUTPUT_DIM])
        all_real_data_splits = tf.split(all_real_data, len(DEVICES), axis=0)

        #DEVICES_A = DEVICES[int(len(DEVICES) / 2):]
        # DEVICES_B = DEVICES[:int(len(DEVICES) / 2)]

        disc_costs = []
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                if ALGORITHM == 'rcgan-u':
                    real_and_fake_data = all_real_data_splits[i]
                else:
                    real_and_fake_data = tf.concat(values=[
                        all_real_data_splits[i],
                        fake_data_splits[i],
                    ], axis=0)
                if ALGORITHM in ["biased", "unbiased"]:
                    real_and_fake_labels = tf.concat(values=[
                        labels_splits[i],
                        labels_random_splits[i],
                    ], axis=0)
                elif ALGORITHM == "rcgan-u": 
                    real_and_fake_labels = labels_splits[i]
                elif ALGORITHM == "rcgan": 
                    real_and_fake_labels = tf.concat(values=[
                        labels_splits[i],
                        labels_biased_splits[i],
                    ], axis=0)                 
                if i == 0:
                    reuse = False
                else:
                    reuse = True

                output, output_wgan  = Discriminator(real_and_fake_data, real_and_fake_labels, update_collection=None, reuse=reuse)
                embedding_y  = Discriminator_projection(real_and_fake_labels, update_collection=None, reuse=reuse)

                if ALGORITHM == "biased" or ALGORITHM == "rcgan":                    
                    disc_all = output_wgan + tf.reshape(tf.reduce_sum(output*embedding_y, axis=1), [-1])    
                    disc_real = disc_all[:int(BATCH_SIZE / len(DEVICES))]
                    disc_fake = disc_all[int(BATCH_SIZE / len(DEVICES)):]
                    if LOSS_TYPE == 'Goodfellow':
                        if SOFT_PLUS:
                            disc_real_l = -tf.reduce_mean(tf.nn.softplus(tf.log(tf.nn.sigmoid(disc_real))))
                            disc_fake_l = -tf.reduce_mean(tf.nn.softplus(tf.log(1 - tf.nn.sigmoid(disc_fake))))
                        else:
                            disc_real_l = -tf.reduce_mean(tf.log(tf.nn.sigmoid(disc_real)))
                            disc_fake_l = -tf.reduce_mean(tf.log(1 - tf.nn.sigmoid(disc_fake)))
                        disc_costs.append(disc_real_l + disc_fake_l)
                    elif LOSS_TYPE == 'HINGE':
                        if SOFT_PLUS:
                            disc_real_l = tf.reduce_mean(tf.nn.softplus(-tf.minimum(0., -1 + disc_real)))
                            disc_fake_l = tf.reduce_mean(tf.nn.softplus(-tf.minimum(0., -1 - disc_fake)))
                        else:
                            disc_real_l = tf.reduce_mean(tf.nn.relu(1. - disc_real))
                            disc_fake_l = tf.reduce_mean(tf.nn.relu(1. + disc_fake))
                        disc_costs.append(disc_real_l + disc_fake_l)
                    elif LOSS_TYPE == 'WGAN':
                        if SOFT_PLUS:
                            disc_costs.append(
                                tf.reduce_mean(tf.nn.softplus(disc_fake)) + tf.reduce_mean(tf.nn.softplus(-disc_real)))
                        else:
                            disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))                
                elif ALGORITHM == "unbiased":
                    disc_real_l_y = [j for j in range(VOCAB_SIZE)]
                    for j in range(VOCAB_SIZE):
                        real_and_fake_labels = tf.concat(values=[
                            tf.convert_to_tensor(j*np.ones((int(BATCH_SIZE / len(DEVICES)),)),tf.int32),
                            labels_random_splits[i],                     
                        ], axis=0)                     
                        embedding_y = Discriminator_projection(real_and_fake_labels, update_collection=None, reuse = True)
                        disc_all = output_wgan + tf.reshape(tf.reduce_sum(output*embedding_y, axis=1), [-1])
                        disc_real = disc_all[:int(BATCH_SIZE / len(DEVICES))]
                        disc_fake = disc_all[int(BATCH_SIZE / len(DEVICES)):]
                        if LOSS_TYPE == 'Goodfellow':
                            if SOFT_PLUS:
                                disc_real_l_y[j] = -tf.reshape(tf.nn.softplus(tf.log(tf.nn.sigmoid(disc_real)))
                                                               ,[int(BATCH_SIZE / len(DEVICES)),1]) 
                                disc_fake_l = -tf.reduce_mean(tf.nn.softplus(tf.log(1 - tf.nn.sigmoid(disc_fake))))
                            else:
                                disc_real_l_y[j] = -tf.reshape(tf.log(tf.nn.sigmoid(disc_real)),[int(BATCH_SIZE / len(DEVICES)),1])
                                disc_fake_l = -tf.reduce_mean(tf.log(1 - tf.nn.sigmoid(disc_fake)))                        
                        elif LOSS_TYPE == 'HINGE':
                            if SOFT_PLUS:
                                disc_real_l_y[j] = tf.reshape(tf.nn.softplus(-tf.minimum(0., -1 + disc_real))
                                                              ,[int(BATCH_SIZE / len(DEVICES)),1])
                                disc_fake_l = tf.reduce_mean(tf.nn.softplus(-tf.minimum(0., -1 - disc_fake)))
                            else:
                                disc_real_l_y[j] = tf.reshape(tf.nn.relu(1. - disc_real),[int(BATCH_SIZE / len(DEVICES)),1])
                                disc_fake_l = tf.reduce_mean(tf.nn.relu(1. + disc_fake))
                        elif LOSS_TYPE == 'WGAN':
                            if SOFT_PLUS:
                                disc_real_l_y[j] = tf.nn.softplus(-disc_real)
                                disc_fake_l = tf.reduce_mean(tf.nn.softplus(disc_fake))
                            else:
                                disc_real_l_y[j] = -disc_real
                                disc_fake_l = tf.reduce_mean(disc_fake)
                    abc = tf.reduce_mean(tf.reduce_sum(tf.concat(disc_real_l_y,1)*labels_inv_weights_splits[i], axis = 1))    
                    disc_costs.append(abc + disc_fake_l)
                elif ALGORITHM == "rcgan-u":
                    disc_real = output_wgan + tf.reshape(tf.reduce_sum(output*embedding_y, axis=1), [-1])

                    output, output_wgan  = Discriminator(fake_data_splits[i], labels_random_splits[i], update_collection=None
                                                         ,reuse = True)
                    fake_labels = tf.convert_to_tensor(np.arange(VOCAB_SIZE), tf.int32)
                    embedding_y = Discriminator_projection(fake_labels, update_collection=None, reuse = True)

                    disc_fake = (
                        tf.expand_dims(output_wgan, 1) + 
                        tf.reduce_sum(tf.expand_dims(output, 1)*
                                      tf.expand_dims(embedding_y, 0), axis=-1))
                    if LOSS_TYPE == 'Goodfellow':
                        if SOFT_PLUS:
                            disc_fake_y = -tf.nn.softplus(tf.log(1. - tf.nn.sigmoid(disc_fake)))
                            disc_real_l = -tf.reduce_mean(tf.nn.softplus(tf.log(tf.nn.sigmoid(disc_real))))
                        else:
                            disc_fake_y = -tf.log(1. - tf.nn.sigmoid(disc_fake))
                            disc_real_l = -tf.reduce_mean(tf.log(tf.nn.sigmoid(disc_real)))
                    elif LOSS_TYPE == 'HINGE':
                        if SOFT_PLUS:
                            disc_fake_y = tf.nn.softplus(-tf.minimum(0., -1 - disc_fake))
                            disc_real_l = tf.reduce_mean(tf.nn.softplus(-tf.minimum(0., -1 + disc_real)))
                        else:
                            disc_fake_y = tf.nn.relu(1. + disc_fake)
                            disc_real_l = tf.reduce_mean(tf.nn.relu(1. - disc_real))
                    elif LOSS_TYPE == 'WGAN':
                        if SOFT_PLUS:
                            disc_fake_y = tf.nn.softplus(disc_fake)
                            disc_real_l = tf.reduce_mean(tf.nn.softplus(-disc_real))
                        else:
                            disc_fake_y = disc_fake
                            disc_real_l = tf.reduce_mean(-disc_real)
                    y_fake_confuse = tf.tensordot(
                        tf.one_hot(labels_random_splits[i], VOCAB_SIZE), confusion_matrix, axes=[[1], [0]])
                    abc = tf.reduce_mean(tf.reduce_sum(disc_fake_y*y_fake_confuse, axis = 1))    
                    disc_costs.append(abc + disc_real_l)
                
                if FLAGS.perm_classifier:
                    if i==0:
                        reuse = False
                    else:
                        reuse = True
                    perm_classifier_real_logits = perm_classifier(all_real_data_splits[i], reuse=reuse)
                    perm_classifier_real_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(
                        perm_classifier_real_logits, tf.one_hot(labels_splits[i], VOCAB_SIZE)))
                    disc_costs[-1] += 1.*perm_classifier_real_loss

        disc_wgan = tf.add_n(disc_costs) / len(DEVICES)
        tf.summary.scalar('D_wgan_cost', disc_wgan)
        disc_cost = disc_wgan
        if DECAY:
            decay = tf.where(
                tf.less(_iteration, 50000),
                tf.maximum(0., 1. - (tf.cast(_iteration, tf.float32) / 100000)), 0.5)
        else:
            decay = 1.
        tf.summary.scalar('lr', LR * decay)

        all_random_labels_G = tf.placeholder(tf.int32, shape=[BATCH_SIZE*GEN_BS_MULTIPLE], name='g1')
        labels_random_splits_G = tf.split(all_random_labels_G, len(DEVICES), axis=0)

        all_labels_biased_G = tf.placeholder(tf.int32, shape=[BATCH_SIZE*GEN_BS_MULTIPLE], name = 'g2')
        labels_biased_splits_G = tf.split(all_labels_biased_G, len(DEVICES), axis=0)

           
        gen_costs = []
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                n_samples = GEN_BS_MULTIPLE * int(BATCH_SIZE / len(DEVICES))
                fake_data_split_G = Generator(n_samples, labels_random_splits_G[i], reuse=True)
                if ALGORITHM == "biased" or ALGORITHM == "unbiased":
                    output, output_wgan = Discriminator(fake_data_split_G,
                                                        labels_random_splits_G[i],
                                                        update_collection="NO_OPS",
                                                        reuse=True)                                             
                    embedding_y  = Discriminator_projection(labels_random_splits_G[i], update_collection=None, reuse=True)
                if ALGORITHM in ["rcgan", "rcgan-u"]:
                    output, output_wgan = Discriminator(fake_data_split_G,
                                                        labels_biased_splits_G[i],
                                                        update_collection="NO_OPS",
                                                        reuse=True)                                                             
                    embedding_y  = Discriminator_projection(labels_biased_splits_G[i], update_collection=None, reuse=True)               


                if ALGORITHM == "rcgan-u":
                    fake_labels = tf.convert_to_tensor(np.arange(VOCAB_SIZE), tf.int32)
                    embedding_y = Discriminator_projection(fake_labels, update_collection=None, reuse = True)
                    disc_fake = (
                        tf.expand_dims(output_wgan, 1) + 
                        tf.reduce_sum(tf.expand_dims(output, 1)*
                                      tf.expand_dims(embedding_y, 0), axis=-1))

                    if LOSS_TYPE == 'Goodfellow':
                        if SOFT_PLUS:
                            disc_fake_y = tf.nn.softplus(-tf.log(tf.nn.sigmoid(disc_fake)))
                        else:
                            disc_fake_y = -tf.log(tf.nn.sigmoid(disc_fake))
                    elif LOSS_TYPE == 'HINGE':
                        if SOFT_PLUS:
                            disc_fake_y = tf.nn.softplus(-disc_fake)
                        else:
                            disc_fake_y = -disc_fake
                    elif LOSS_TYPE == 'WGAN':
                        if SOFT_PLUS:
                            disc_fake_y = tf.nn.softplus(-disc_fake)
                        else:
                            disc_fake_y = -disc_fake
                    y_fake_confuse = tf.tensordot(
                      tf.one_hot(labels_random_splits_G[i], VOCAB_SIZE), confusion_matrix, axes=[[1], [0]])
                    abc = tf.reduce_mean(tf.reduce_sum(disc_fake_y*y_fake_confuse, axis = 1))    
                    gen_costs.append(abc)

                else:
                    disc_fake =  output_wgan + tf.reshape(tf.reduce_sum(output*embedding_y, axis=1), [-1])   
                    if LOSS_TYPE == 'Goodfellow':
                        if SOFT_PLUS:
                            gen_costs.append(tf.reduce_mean(tf.nn.softplus(-tf.log(tf.nn.sigmoid(disc_fake)))))
                        else:
                            gen_costs.append(-tf.reduce_mean(tf.log(tf.nn.sigmoid(disc_fake))))
                    elif LOSS_TYPE == 'HINGE':
                        if SOFT_PLUS:
                            gen_costs.append(tf.reduce_mean(tf.nn.softplus(-disc_fake)))
                        else:
                            gen_costs.append(-tf.reduce_mean(disc_fake))
                    elif LOSS_TYPE == 'WGAN':
                        if SOFT_PLUS:
                            gen_costs.append(tf.reduce_mean(tf.nn.softplus(-disc_fake)))
                        else:
                            gen_costs.append(-tf.reduce_mean(disc_fake))

                if FLAGS.perm_classifier:
                    perm_classifier_fake_logits = perm_classifier(fake_data_split_G, reuse=True)
                    perm_classifier_fake_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(
                        perm_classifier_fake_logits, tf.one_hot(labels_random_splits_G[i], VOCAB_SIZE)))
                    gen_costs[-1] += FLAGS.perm_multiplier*perm_classifier_fake_loss

        gen_cost = (tf.add_n(gen_costs) / len(DEVICES))
        tf.summary.scalar('G_wgan_cost', gen_cost)
        gen_params = [var for var in tf.trainable_variables() if 'Generator' in var.name]
        logging.debug('\ngen_params:')
        for var in gen_params:
            logging.debug(var.name)

        disc_params = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]        
        logging.debug('\ndisc_params:')
        for var in disc_params:
            logging.debug(var.name)

        logging.debug('\ntrainable_variables.name:')
        for var in tf.trainable_variables():
            logging.debug(var.name)

        disc_opt = tf.train.AdamOptimizer(learning_rate=LR * decay, beta1=0., beta2=0.9)
        disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
        disc_train_op = disc_opt.apply_gradients(disc_gv)

        gen_opt = tf.train.AdamOptimizer(learning_rate=LR * decay, beta1=0., beta2=0.9)
        gen_gv = gen_opt.compute_gradients(gen_cost, var_list=gen_params)
        gen_train_op = gen_opt.apply_gradients(gen_gv)

        confuse_train_op = tf.no_op()
        if ALGORITHM == 'rcgan-u':
            if FLAGS.confuse_lr_decay:
                confuse_lr = LR * decay * FLAGS.confuse_multiplier
            else:
                confuse_lr = LR * FLAGS.confuse_multiplier
            confuse_train_op = tf.train.AdamOptimizer(confuse_lr, beta1=0., beta2=0.9) \
                      .minimize(gen_cost, var_list=[confusion_logits])


        # Function for generating samples
        frame_i = [0]
        fixed_noise = tf.constant(np.random.normal(size=(100, Z_DIM)).astype('float32'))
        # airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
        sample_labels = [[k]*10 for k in range(10)]
        sample_labels = [k for item in sample_labels for k in item]
        fixed_labels = tf.constant(np.array(sample_labels, dtype='int32'))#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10
        fixed_noise_samples = Generator(100, fixed_labels, noise=fixed_noise, reuse=True)

        def generate_image(frame, true_dist):
            samples = session.run(fixed_noise_samples)
            samples = ((samples + 1.) * (255. / 2)).astype('int32')
            common.misc.save_images(samples.reshape((100, IMG_SIZE, IMG_SIZE, IMG_DIM)),
                                    os.path.join(DIR, 'samples_{}.png'.format(frame)))

        # Function for calculating inception score
        fake_labels_100 = tf.cast(tf.random_uniform([100]) * 10, tf.int32)
        samples_100 = Generator(100, fake_labels_100, reuse=True)
        def get_inception_score(n):
            # For inception_score_new2
            all_samples = []
            for i in range(int(n / 100)):
                all_samples.append(session.run(samples_100))
            all_samples = np.concatenate(all_samples, axis=0)
            all_samples = all_samples.reshape((-1, IMG_SIZE, IMG_SIZE, IMG_DIM)).transpose(0, 3, 1, 2)
            return common.inception.inception_score_.get_inception_score(all_samples)

        label_100_list = [label for label in range(10) for _ in range(10)]
        fake_deterministic_labels_100 = tf.cast(tf.constant(label_100_list), tf.int32)
        deterministic_samples_100 = Generator(100, fake_deterministic_labels_100, reuse=True)
        def save_samples(n):
            all_samples = []
            all_labels = []
            for i in range(int(n / 100)):
                all_samples.append(session.run(deterministic_samples_100))
                all_labels.append(label_100_list)
            all_samples = np.concatenate(all_samples, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            all_samples = ((all_samples + 1.) * (255.99 / 2)).astype('int32')
            # all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
            all_samples = all_samples.reshape((-1, IMG_SIZE, IMG_SIZE, IMG_DIM))
            return all_samples, all_labels
            
        # Function for reading data
        train_gen, dev_gen = dataset_.load(BATCH_SIZE, DATA_DIR, C_ALPHA)
        def inf_train_gen():
            while True:
                for images_, labels_, labels_random_, labels_biased_, labels_inv_weights_ in train_gen():
                    yield images_, labels_, labels_random_, labels_biased_, labels_inv_weights_
        def inf_train_gen_G():          
            _generator = train_gen()
            while True:
                labels_random_list = []
                labels_biased_list = []
                for _ in range(GEN_BS_MULTIPLE):                                                      
                    try:
                        _, _, labels_random_list_element, labels_biased_list_element, _ = _generator.__next__()
                    except StopIteration:
                        _generator = train_gen()
                        _, _, labels_random_list_element, labels_biased_list_element, _ = _generator.__next__()
                    labels_random_list.append(labels_random_list_element)
                    labels_biased_list.append(labels_biased_list_element)
                yield (np.concatenate(labels_random_list, axis= 0), np.concatenate(labels_biased_list, axis = 0))

        gen = inf_train_gen()
        gen_G = inf_train_gen_G()

        for name, grads_and_vars in [('G', gen_gv), ('D', disc_gv)]:
            logging.debug("{} Params:".format(name))
            total_param_count = 0
            for g, v in grads_and_vars:
                shape = v.get_shape()
                shape_str = ",".join([str(x) for x in v.get_shape()])

                param_count = 1
                for dim in shape:
                    param_count *= int(dim)
                total_param_count += param_count

                if g is None:
                    logging.debug("\t{} ({}) [no grad!]".format(v.name, shape_str))
                else:
                    logging.debug("\t{} ({})".format(v.name, shape_str))
            logging.debug("Total param count: {}".format(locale.format("%d", total_param_count, grouping=True)))

        summaries_op = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=5)
        summary_writer = tf.summary.FileWriter(CHECKPOINT_DIR, graph=session.graph)
        session.run(tf.global_variables_initializer())

        if RESTORE:
            ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
            if ckpt:
                logging.info('restore model from: {}...'.format(ckpt))
                saver.restore(session, ckpt)
                
        _random_labels_G, _labels_biased_G = next(gen_G)
        inception_score_max = 0.0
        gen_label_acc_max = 0.0
        for iteration in range(ITERS):
            start_time = time.time()

            if ALGORITHM == 'rcgan-u' and (iteration%100==0 or iteration < 500):
                logging.debug('confusion_matrix: ')
                np.set_printoptions(precision=3, suppress=True)
                logging.debug('\n{}'.format(session.run(confusion_matrix)))
                np.set_printoptions()

            if 0 < iteration:
                _random_labels_G, _labels_biased_G = next(gen_G)
                #logging.debug('test1: {}'.format(_random_labels_G))
                _ = session.run([gen_train_op, confuse_train_op], 
                    feed_dict={_iteration: iteration,                            
                               all_random_labels_G: _random_labels_G,
                               all_labels_biased_G: _labels_biased_G})
            
            for i in range(N_CRITIC):
                _data, _labels, _random_labels, _labels_biased, _labels_inv_weights = next(gen)
                _disc_cost, _disc_wgan, _gen_cost, _, summaries = session.run(
                    [disc_cost, disc_wgan, gen_cost, disc_train_op, summaries_op],
                    feed_dict={all_real_data_int: _data,
                               all_real_labels: _labels,
                               all_random_labels: _random_labels,
                               all_labels_biased: _labels_biased,
                               all_labels_inv_weights: _labels_inv_weights, 
                               all_random_labels_G: _random_labels_G,
                               all_labels_biased_G: _labels_biased_G,                           
                               _iteration: iteration})

            summary_writer.add_summary(summaries, global_step=iteration)

            # lib.plot.plot('cost', _disc_cost)
            lib.plot.plot('d_cost', _disc_wgan)
            lib.plot.plot('g_cost', _gen_cost)
            if CONDITIONAL and ACGAN:
                lib.plot.plot('disc_wgan', _disc_wgan)
                lib.plot.plot('acgan', _disc_acgan)
                lib.plot.plot('acc_real', _disc_acgan_acc)
                lib.plot.plot('acc_fake', _disc_acgan_fake_acc)

            if iteration % INCEPTION_FREQUENCY == INCEPTION_FREQUENCY - 1:
                logging.info('starting inception score computation.')
                inception_score = get_inception_score(50000)
                inception_score_max = max(inception_score_max, inception_score[0])
                lib.plot.plot('inception_50k', inception_score[0])
                lib.plot.plot('inception_50k_std', inception_score[1])
                lib.plot.plot('inception_50k_max', inception_score_max)
                logging.info('finished inception score computation.')

            if SAMPLE_SAVE_FREQUENCY and iteration % SAMPLE_SAVE_FREQUENCY == SAMPLE_SAVE_FREQUENCY - 1:
                logging.info('starting saving samples.')
                samples_for_save, _ = save_samples(10000)
                np.save(os.path.join(DIR, '_samples_{}'.format(iteration)), samples_for_save)
                logging.info('finished saving samples.')
                
            # Calculate dev loss and generate samples every 100 iters
            if iteration % SAMPLE_FREQUENCY == SAMPLE_FREQUENCY - 1:
                logging.info('starting calculating dev cost.')
                dev_disc_costs = []
                for images, _labels, _random_labels, _labels_biased, _labels_inv_weights in dev_gen():
                    _dev_disc_cost = session.run([disc_cost],
                                                 feed_dict={all_real_data_int: images,
                                                            all_real_labels: _labels,
                                                            all_random_labels: _random_labels,
                                                            all_labels_biased: _labels_biased,
                                                            all_labels_inv_weights: _labels_inv_weights,
                                                           })                                                                    
                    dev_disc_costs.append(_dev_disc_cost)
                lib.plot.plot('dev_cost', np.mean(dev_disc_costs))
                logging.info('finished calculating dev cost.')

                logging.info('starting generating samples.')
                generate_image(iteration, _data)
                logging.info('finished generating samples.')

            if iteration % GENERATED_LABEL_ACCURACY_FREQ == GENERATED_LABEL_ACCURACY_FREQ  - 1:
                logging.info('starting calculating generated label accuracy.')

                generated_samples, generate_labels = save_samples(1000)
                accuracy = generated_label_accuracy(generated_samples, generate_labels)
                if gen_label_acc_max < accuracy:
                    gen_label_acc_max = accuracy
                
                lib.plot.plot('gen_label_acc', accuracy)
                lib.plot.plot('gen_label_acc_max', gen_label_acc_max)
                logging.info('finished calculating generated label accuracy.')

            if (iteration < 500) or (iteration % 1000 == 999):
                logging.info('start flushing plots and checpoints.')
                lib.plot.dir_flush(DIR)

                if not os.path.exists(CHECKPOINT_DIR):
                    os.mkdir(CHECKPOINT_DIR)
                saver.save(session, os.path.join(CHECKPOINT_DIR, 'model.ckpt'), global_step=iteration)
                logging.info('finished flushing plots and checpoints.')

            lib.plot.tick()

        if ITERS:
            summary_writer.flush()
            summary_writer.close()

        if FLAGS.perm_gen_label_acc:
            logging.info('starting calculating min. permuted generated label accuracy.')
            generated_samples, generate_labels = save_samples(1000)
            accuracy = generated_label_accuracy(
                generated_samples, generate_labels,
                confusion_matrix=session.run(confusion_matrix))
            lib.plot.plot('gen_label_acc', accuracy)
            logging.info('finished calculating min. permuted generated label accuracy.')
        else:
            logging.info('starting calculating generated label accuracy.')
            generated_samples, generate_labels = save_samples(1000)
            accuracy = generated_label_accuracy(generated_samples, generate_labels)
            lib.plot.plot('gen_label_acc', accuracy)
            logging.info('finished calculating generated label accuracy.')


if __name__ == '__main__':
    tf.app.run()