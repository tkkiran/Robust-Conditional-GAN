import os
import scipy.misc
import numpy as np
from datetime import datetime

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables
import utils

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 5, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 100, "The size of batch images")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [mnist]")
flags.DEFINE_string("checkpoint_dir", "rcgan", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("checkpoint", None, "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples/", "Directory name to save the image samples")
flags.DEFINE_string("data_dir", "../data/", "Root directory of dataset [data]")
flags.DEFINE_string('dir_prefix', None, "dir name prefix")
flags.DEFINE_string('logs_dir', './logs', "logs directory")
flags.DEFINE_boolean('logs_at_ckpt', False, "set logs dir to chechkpoint dir")
flags.DEFINE_string('script_file', None, 
                    "script file name for storing script along with results")

flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("z_dim", 100, "Dimension of input noise Z to the generator")

flags.DEFINE_string("algorithm", "biased", "[biased, unbiased, rcgan, ambient]")
flags.DEFINE_boolean("estimate_confuse", True, "whether to estimate confusion matrix")
flags.DEFINE_float("confuse_multiplier", 10.0, "learning rate multiplier for confusion matrix")
flags.DEFINE_boolean("perm_regularizer", True,
                     "whether to use auxillary permutation regularizer classifier")
flags.DEFINE_float("perm_multiplier", 10.0, "learning rate multiplier for permutation regularizer")

flags.DEFINE_float("alpha", 1.0, "noise in labels")
flags.DEFINE_boolean(
  "confusion_class_depend", False, 
  "whether to generate rows of confusion matrix in a class dependent way or in one coin model way")

flags.DEFINE_string("disc_type", "vanilla", "type of discriminator to use [vanilla, projection]")
flags.DEFINE_string('loss_fn', 'hinge', 'GAN loss function')
flags.DEFINE_boolean("real_match", False, 'whether to match y_gen with y_real in for each batch')

flags.DEFINE_boolean('add_noise', False, 'whether to add noise to both real and fake labels y_real, y_fake')
flags.DEFINE_float("noise_alpha", 0.3, "effective noise in labels")
flags.DEFINE_integer("noise_start", 30, "noise schedule start")
flags.DEFINE_integer("noise_end", 80, "noise schedule end")

flags.DEFINE_boolean('concat_y', False, 'whether to concat y to projection discriminator')
flags.DEFINE_list('concat_y_layers', ['1',], 'layers of projection discriminator where we want to concat y [1, 2, 3, 4]')
flags.DEFINE_boolean('spectral_norm', True, 'whether to use spectral normalization on conv2d layers of the discriminator')
flags.DEFINE_boolean('max_norm', True, 'whether to use maximum value (clip) normalization on linear layers of discriminator')

flags.DEFINE_integer("recover_epoch", 1000, "Epoch to train [25]")
flags.DEFINE_integer("recover_batch_size", 500, "The size of batch images [64]")
flags.DEFINE_float("recover_learning_rate", 5.e+2, "Learning rate of for adam [0.0002]")
FLAGS = flags.FLAGS


def main(_):
  FLAGS.concat_y_layers = [int(x) for x in FLAGS.concat_y_layers]

  if FLAGS.dir_prefix is None:
    FLAGS.dir_prefix = ''
  else:
    FLAGS.dir_prefix = FLAGS.dir_prefix + '_'

  if FLAGS.checkpoint is None:
    FLAGS.checkpoint_dir = os.path.join(
      FLAGS.checkpoint_dir, FLAGS.dir_prefix + FLAGS.algorithm + "_" + str(FLAGS.alpha) + "_" + FLAGS.disc_type + 
      "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
  else:
    FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.checkpoint)
  FLAGS.sample_dir = os.path.join(FLAGS.checkpoint_dir, 'samples/') 

  pp.pprint(flags.FLAGS.__flags)
  FLAGS.input_height = 28
  FLAGS.output_height = 28
  if FLAGS.input_width is None:
    FLAGS.input_width = 28#FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = 28#FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  file_list = ['main.py', 'model.py', 'utils.py', 'ops.py', 'sn.py']
  utils.dump_script(FLAGS.checkpoint_dir, FLAGS.script_file, file_list=file_list)

  if FLAGS.logs_at_ckpt:
    FLAGS.logs_dir = FLAGS.checkpoint_dir

  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  FLAGS.dataset = 'mnist'
  with tf.Session(config=run_config) as sess:
    if FLAGS.dataset == 'mnist':
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          y_dim=10,
          z_dim=FLAGS.z_dim,
          dataset_name=FLAGS.dataset,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          data_dir=FLAGS.data_dir,
          algorithm=FLAGS.algorithm,
          estimate_confuse=FLAGS.estimate_confuse,
          perm_regularizer=FLAGS.perm_regularizer,
          alpha=FLAGS.alpha,
          disc_type=FLAGS.disc_type,
          add_noise=FLAGS.add_noise,
          noise_alpha=FLAGS.noise_alpha,
          config=FLAGS)

    show_all_variables()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        print("[!] Training a model first, then run test mode")
        dcgan.train(FLAGS)
    
    dcgan.recover_labels(FLAGS)

if __name__ == '__main__':
  tf.app.run()
