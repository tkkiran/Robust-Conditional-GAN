from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *
import utils


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         checkpoint_dir=None, sample_dir=None, 
         data_dir='./data',algorithm="biased", estimate_confuse=False,
         perm_regularizer=False, alpha=1.0, disc_type="vanilla",
         add_noise=False, noise_alpha=1.0, config=None):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.algorithm = algorithm
    self.estimate_confuse = estimate_confuse
    self.perm_regularizer = perm_regularizer
    self.alpha = alpha
    self.disc_type = disc_type

    self.add_noise = add_noise
    self.noise_alpha = noise_alpha
        
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.config = config

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    if self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.checkpoint_dir = checkpoint_dir
    self.data_dir = data_dir

    if self.dataset_name == 'mnist':
      self.data_X, self.data_y_actual, self.data_y_real, self.data_y_gen, self.data_y_fake, self.data_y_real_weights  = self.load_mnist()
      self.c_dim = self.data_X[0].shape[-1]

    self.grayscale = (self.c_dim == 1)

    self.build_model()

  def build_model(self):
    if self.y_dim:
      self.y_real = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y_real')
      self.y_fake = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y_fake')
      self.y_gen = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y_gen')
      self.y_real_weights = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y_real_weights')
      if self.estimate_confuse:
        self.confusion_logits = tf.get_variable(
          'confusion_logits', dtype=tf.float32, shape=[self.y_dim, self.y_dim],
          trainable=True)
        self.confusion_matrix = tf.nn.softmax(self.confusion_logits, dim=-1)
      else:
        self.confusion_matrix = tf.constant(self.confusion_matrix_actual.astype(np.float32))
    else:
      self.y = None

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)
        
    self.G                  = self.generator(self.z, self.y_gen)
    self.sampler            = self.gen_sampler(self.z, self.y_gen)

    if self.config.loss_fn == 'hinge':
      d_loss_real_fn = lambda x: tf.nn.relu(1 - x)
      d_loss_fake_fn = lambda x: tf.nn.relu(1 + x)
      g_loss_fn = lambda x: -x
    elif self.config.loss_fn == 'ce':
      d_loss_real_fn = lambda x: sigmoid_cross_entropy_with_logits(
        x, tf.ones_like(x))
      d_loss_fake_fn = lambda x: sigmoid_cross_entropy_with_logits(
        x, tf.zeros_like(x))
      g_loss_fn = lambda x: sigmoid_cross_entropy_with_logits(
        x, tf.ones_like(x))
    else:
      raise ValueError('Unknown self.config.loss_fn: {}!'.format(self.config.loss_fn))

    ##Real data
    if self.algorithm in ["biased", "rcgan", "ambient"]:
        self.D, self.D_logits   = self.discriminator(inputs, self.y_real, reuse=False)
        self.d_loss_real = tf.reduce_mean(d_loss_real_fn(self.D_logits))    
    elif self.algorithm == "unbiased":
        self.D_all = [i for i in range(10)]
        self.D_logits_all = [i for i in range(10)]
        self.d_loss_real_all = [i for i in range(10)]
        for i in range(10):
            np_labels_i = np.zeros((1,10))
            np_labels_i[0,i] = 1
            labels_i = tf.matmul(tf.ones([self.batch_size,1], tf.float32), tf.convert_to_tensor(np_labels_i,tf.float32))
            if i ==0:
                reuse = False
            else:
                reuse = True
            self.D_all[i], self.D_logits_all[i] = self.discriminator(inputs, labels_i, reuse=reuse)
            self.d_loss_real_all[i] = d_loss_real_fn(self.D_logits_all[i])     

        self.D_all = tf.concat(self.D_all,1)        
        self.D_logits_all = tf.concat(self.D_logits_all,1)
        self.d_loss_real_all = tf.concat(self.d_loss_real_all,1)
        
        self.D = tf.reduce_sum(self.D_all*self.y_real_weights, axis = 1)
        self.D_logits = tf.reduce_sum(self.D_logits_all*self.y_real_weights, axis = 1)
        self.d_loss_real = tf.reduce_mean(tf.reduce_sum(self.d_loss_real_all*self.y_real_weights, axis = 1))

    ##Generator created data 
    self.d_loss_fake = None
    self.g_loss = None
    if self.algorithm in ["rcgan", "ambient"]:
      if not self.estimate_confuse:
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y_fake, reuse=True)
      else:
        self.D_all_ = [i for i in range(10)]
        self.D_logits_all_ = [i for i in range(10)]
        self.d_loss_fake_all_ = [i for i in range(10)]
        self.g_loss_all = [i for i in range(10)]
        for i in range(10):
          np_labels_i = np.zeros((1,10))
          np_labels_i[0,i] = 1
          labels_i = tf.matmul(tf.ones([self.batch_size,1], tf.float32), tf.convert_to_tensor(np_labels_i,tf.float32))
          self.D_all_[i], self.D_logits_all_[i] = self.discriminator(self.G, labels_i, reuse=True)
          self.d_loss_fake_all_[i] = d_loss_fake_fn(self.D_logits_all_[i])     
          self.g_loss_all[i] = g_loss_fn(self.D_logits_all_[i])
        self.D_all_ = tf.concat(self.D_all_,1)        
        self.D_logits_all_ = tf.concat(self.D_logits_all_,1)
        self.d_loss_fake_all_ = tf.concat(self.d_loss_fake_all_,1)
        self.g_loss_all = tf.concat(self.g_loss_all,1)

        y_fake_confuse = tf.tensordot(
          self.y_gen, self.confusion_matrix, axes=[[1], [0]])
        self.D_ = tf.reduce_sum(self.D_all_*y_fake_confuse, axis = 1)
        self.D_logits_ = tf.reduce_sum(self.D_logits_all_*y_fake_confuse, axis = 1)
        self.d_loss_fake = tf.reduce_mean(tf.reduce_sum(self.d_loss_fake_all_*y_fake_confuse, axis = 1))
        self.g_loss = tf.reduce_mean(tf.reduce_sum(self.g_loss_all*y_fake_confuse, axis = 1))

    elif self.algorithm in ["biased", "unbiased"]:
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y_gen, reuse=True)

    if self.d_loss_fake is None:
      self.d_loss_fake = tf.reduce_mean(d_loss_fake_fn(self.D_logits_))
    if self.g_loss is None:
      self.g_loss = tf.reduce_mean(g_loss_fn(self.D_logits_))

    if self.perm_regularizer:
      self.classifier_logits = self.classifier(inputs)
      self.classifier_logits_ = self.classifier(self.G, reuse=True)

      self.class_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(
        self.classifier_logits, self.y_real))
      self.class_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(
        self.classifier_logits_, self.y_gen))
    else:
      self.class_loss_real = tf.constant(0.0, dtype=tf.float32)
      self.class_loss_fake = tf.constant(0.0, dtype=tf.float32)
        
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)    

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)
    self.class_loss_real_sum = scalar_summary(
      "class_loss_real", self.class_loss_real)
    self.class_loss_fake_sum = scalar_summary(
      "class_loss_fake", self.class_loss_fake)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(
                self.d_loss + 1.*self.class_loss_real, 
                var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(
                self.g_loss + config.perm_multiplier*self.class_loss_fake,
                var_list=self.g_vars)
    c_optim = tf.no_op()
    if self.estimate_confuse:
        c_optim = tf.train.AdamOptimizer(
            config.learning_rate*config.confuse_multiplier, beta1=config.beta1) \
                  .minimize(self.g_loss, var_list=[self.confusion_logits])
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum, self.class_loss_fake_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum, self.class_loss_real_sum])
    self.writer = SummaryWriter(self.config.logs_dir, self.sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    
    if config.dataset == 'mnist':
      samples = []
      for i in range(10):
        samples.append(np.where(self.data_y_gen[:,i]==1)[0][0:10])
      samples = [i for j in samples for i in j]
      sample_inputs = self.data_X[samples[0:100]]
      sample_labels = self.data_y_gen[samples[0:100]]
  
    counter = 1
    start_time = time.time()

    for epoch in range(config.epoch):
      if config.dataset == 'mnist':
        batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
      indices = [i for i in range(len(self.data_X))] 
      #random.shuffle(indices)

      if self.add_noise:
        alpha_start = ((self.noise_alpha - (1. - self.alpha)/(self.y_dim-1))/
                       (self.alpha - (1. - self.alpha)/(self.y_dim-1)))
        alpha_start = min(1.0, alpha_start)

        if self.noise_alpha > 0.9:
          raise ValueError('same rate activated, but effective noise alpha {} > 0.9!'.format(self.noise_alpha))

        if alpha_start == 1.:
          end_epoch = self.config.noise_start
        else:
          end_epoch = self.config.noise_start + (
            (self.config.noise_end - self.config.noise_start)/
            (0.9 - self.noise_alpha)*(self.alpha - self.noise_alpha))
          end_epoch = min(self.config.noise_end, end_epoch)

        if epoch < self.config.noise_start:
          noise_alpha = alpha_start
        elif epoch < end_epoch:
          noise_alpha = (
            alpha_start + 
            (1.-alpha_start)*(epoch - self.config.noise_start)/
            (end_epoch - self.config.noise_start))
        else:
          noise_alpha = 1.0
        noise_alpha = min(1.0, noise_alpha)

        # One-coin confusion matrix
        noise_C = ((1-noise_alpha)/(self.y_dim-1))*np.ones((self.y_dim,self.y_dim)) + (noise_alpha - (1-noise_alpha)/(self.y_dim-1))*np.eye((self.y_dim)) 

        data_y_real_orig = self.data_y_real
        data_y_fake_orig = self.data_y_fake

        self.data_y_real = np.zeros_like(self.data_y_real)
        self.data_y_fake = np.zeros_like(self.data_y_fake)

        for ii, _ in enumerate(self.data_y_real):
          self.data_y_real[ii] = np.random.multinomial(
            1, noise_C[np.argmax(data_y_real_orig[ii]),:], size=1)
          self.data_y_fake[ii] = np.random.multinomial(
            1, noise_C[np.argmax(data_y_fake_orig[ii]),:], size=1)

      for idx in range(0, batch_idxs):
        if config.dataset == 'mnist':
          batch_images = self.data_X[indices[idx*config.batch_size:(idx+1)*config.batch_size]]
          batch_labels_real = self.data_y_real[indices[idx*config.batch_size:(idx+1)*config.batch_size]]
          batch_labels_gen = self.data_y_gen[indices[idx*config.batch_size:(idx+1)*config.batch_size]]  
          batch_labels_fake = self.data_y_fake[indices[idx*config.batch_size:(idx+1)*config.batch_size]] 
          batch_labels_real_weights = self.data_y_real_weights[indices[idx*config.batch_size:(idx+1)*config.batch_size]] 
        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)

        if config.dataset == 'mnist':
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ 
              self.inputs: batch_images,
              self.z: batch_z,
              self.y_real: batch_labels_real,
              self.y_gen: batch_labels_gen,
              self.y_fake: batch_labels_fake,
              self.y_real_weights: batch_labels_real_weights
            })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, _, summary_str = self.sess.run([g_optim, c_optim, self.g_sum],
            feed_dict={
              self.z: batch_z, 
              self.y_gen: batch_labels_gen,
              self.y_fake: batch_labels_fake,
            })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, _, summary_str = self.sess.run([g_optim, c_optim, self.g_sum],
            feed_dict={ self.z: batch_z,               
              self.y_gen: batch_labels_gen,
              self.y_fake: batch_labels_fake, })
          self.writer.add_summary(summary_str, counter)
          
          errD_fake = self.d_loss_fake.eval({
              self.z: batch_z, 
              self.y_gen: batch_labels_gen,
              self.y_fake: batch_labels_fake,
          })
          errD_real = self.d_loss_real.eval({
              self.inputs: batch_images,
              self.y_real: batch_labels_real,
              self.y_real_weights: batch_labels_real_weights
          })
          errG = self.g_loss.eval({
              self.z: batch_z,
              self.y_gen: batch_labels_gen,
              self.y_fake: batch_labels_fake,
          })
          prob_fake = self.D_.eval({
              self.z: batch_z,
              self.y_gen: batch_labels_gen,
              self.y_fake: batch_labels_fake,
          })
          prob_real = self.D.eval({
              self.inputs: batch_images,
              self.y_real: batch_labels_real,
              self.y_real_weights: batch_labels_real_weights
          })

          if self.estimate_confuse:
            confusion_l1_norm = np.mean(np.sum(np.abs(
              self.confusion_matrix_actual - self.sess.run(self.confusion_matrix)), axis=1))
            diag_idx = list(range(self.y_dim))
            diag_diff = np.abs(
              self.confusion_matrix_actual[diag_idx,diag_idx] - 
              self.sess.run(self.confusion_matrix)[diag_idx, diag_idx])
            diag_actual = self.confusion_matrix_actual[0,0]
            

          if self.perm_regularizer:
            err_class_real, err_class_fake = self.sess.run(
              [self.class_loss_real, self.class_loss_fake], 
              feed_dict={
                self.z: batch_z,
                self.inputs: batch_images,
                self.y_real: batch_labels_real,
                self.y_gen: batch_labels_gen,
                self.y_fake: batch_labels_fake,})

        counter += 1
        correct_real = len(np.where(prob_real >= 0.5)[0])
        correct_fake = len(np.where(prob_fake <= 0.5)[0])
        
        if (epoch < 1 and idx < 20) or idx%350 == 0:
          print("Epoch: [%2d] [%4d/%4d] time: %4.2f, d_loss: %.3f, g_loss: %.3f, "#"class_real: %.3f, class_fake: %.3f\n"
                "d_real: %2d, %.3f, %.3f, d_fake: %2d, %.3f, %.3f" 
            % (epoch, idx, batch_idxs, \
               time.time() - start_time, errD_fake+errD_real, errG, \
               #err_class_real, err_class_fake, \
               correct_real, min(prob_real), \
               max(prob_real), correct_fake, min(prob_fake), max(prob_fake)))

        confusion_matrix_freq = 0
        if self.estimate_confuse and confusion_matrix_freq and np.mod(counter, confusion_matrix_freq*700) == 1:
          if ( 
              True
#               (idx<100) or (idx>=100 and idx%5 == 0)
             ):
            np.set_printoptions(precision=3, suppress=True)
            print('confuse_matrix=')
            print(self.sess.run(self.confusion_matrix))
            np.set_printoptions()

        gen_sample_freq = 1 #5
        if np.mod(counter, gen_sample_freq*700) == 1:
          if config.dataset == 'mnist': 
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
              self.y_real: sample_labels,
              self.y_gen: sample_labels,
              self.y_fake: sample_labels,
              self.y_real_weights: sample_labels
              }
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            if False:
              if not (epoch == gen_sample_freq-1 or (epoch+1)%20 == 0):
               os.remove('./{}/train_{:02d}_{:04d}.png'.format(
                config.sample_dir, epoch-gen_sample_freq, idx)) 

            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
        if np.mod(counter, 700) == 1:
          self.save(config.checkpoint_dir, counter)

      if self.add_noise:
        self.data_y_real = data_y_real_orig
        self.data_y_fake = data_y_fake_orig

      sample_freq = 5
      if np.mod(epoch+1, sample_freq) == 0:
        samples = []
        for i in range(100):
            sample_z = np.random.uniform(-1, 1, size=(100 , self.z_dim))        
            samples.append(self.sess.run(
              self.sampler,
              feed_dict={
              self.z: sample_z,
              self.y_gen: sample_labels,
              }
            ))
        np.save(config.sample_dir+ "/samples_" + str(epoch), samples)
        if epoch+1 != sample_freq:
          os.remove(config.sample_dir+ "/samples_" + str(epoch-sample_freq) + '.npy')
        generated_label_acc = utils.generated_label_accuracy(
          self.dataset_name, np.array(samples))
        print('######EPOCH={}, mean generated label accuracy={}'.format(
          epoch, generated_label_acc))


  def recover_labels(self, config):
    from datetime import datetime
       
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      raise Exception(" [!] Load failed...")
    intialized_variables = set(tf.all_variables()) # https://stackoverflow.com/a/35618160

    self.batch_size = config.recover_batch_size
    
    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.sample_actual = tf.placeholder(
      tf.float32, [None] + image_dims, name='sample_actual')
    self.y_actual = tf.placeholder(
      tf.float32, [config.recover_batch_size, self.y_dim], name='y_actual')
   
    bignum = 1
    self.y_logit_recover = tf.get_variable(
      'y_logit_recover', dtype=tf.float32, 
      shape=[config.recover_batch_size, self.y_dim], trainable=True)
    self.y_recover = tf.nn.softmax(bignum*self.y_logit_recover, dim=-1)
    
    self.batch_size = config.recover_batch_size*self.y_dim

    hard_y_recover = tf.tile(tf.constant(np.eye(self.y_dim), dtype=tf.float32),
      [config.recover_batch_size, 1])

    self.z_recover = tf.get_variable(
      'z_recover', dtype=tf.float32,
      shape=[config.recover_batch_size*self.y_dim, self.z_dim], trainable=True)
    z_recover_tile = self.z_recover
    
    sample_recover_each_y = self.gen_sampler(z_recover_tile, hard_y_recover)
    sample_recover_each_y = tf.reshape(
      tf.expand_dims(sample_recover_each_y, axis=1),
      [config.recover_batch_size, self.y_dim]  + image_dims)
    
    sq_sum = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(
      (tf.expand_dims(self.sample_actual, axis=1) - sample_recover_each_y)**2,
      axis=-1), axis=-1), axis=-1)
    self.mse_loss = tf.reduce_mean(tf.reduce_sum(sq_sum*self.y_recover, axis=-1))

    mse_loss_summ = scalar_summary("mse_loss", self.mse_loss)

    y_onehot_recover = tf.one_hot(tf.argmax(self.y_recover, axis=-1), self.y_dim)
    self.zero_one_loss = tf.losses.cosine_distance(self.y_actual, y_onehot_recover, dim=-1)
    zero_one_loss_summ = scalar_summary("zero_one_loss", self.zero_one_loss)    
    summary_list = [mse_loss_summ, zero_one_loss_summ]

    nwrong = 15
    wrong_images_idx = tf.nn.top_k(
      tf.reduce_sum(tf.abs(self.y_recover - self.y_actual), axis=-1), k=nwrong).indices

    bar_graph = np.zeros([100, 100], dtype=np.float32)
    ii_lower = np.tril_indices_from(bar_graph)
    bar_graph[ii_lower] = 1.0
    bar_graph = tf.constant(np.flip(bar_graph, axis=-1)[
      np.newaxis,:,:,np.newaxis])

    wrong_y_recover = tf.minimum(tf.constant([99]), tf.maximum(tf.constant([0]), 
      tf.cast(tf.gather(self.y_recover, wrong_images_idx)*100, dtype=tf.int32)))
    wrong_y_actual = tf.minimum(tf.constant([99]), tf.maximum(tf.constant([0]), 
      tf.cast(tf.gather(self.y_actual, wrong_images_idx)*100, dtype=tf.int32)))
    wrong_bar_recover = []
    wrong_bar_actual = []
    for i in range(nwrong):
      wrong_bar_recover.append(tf.image.resize_images(
        tf.gather(bar_graph, wrong_y_recover[i], axis=2), 
        size=[28, 50],
        method=tf.image.ResizeMethod.AREA))
      wrong_bar_actual.append(tf.image.resize_images(
        tf.gather(bar_graph, wrong_y_actual[i], axis=2), 
        size=[28, 50],
        method=tf.image.ResizeMethod.AREA))
    wrong_bar_recover = tf.concat(wrong_bar_recover, axis=1)
    wrong_bar_actual = tf.concat(wrong_bar_actual, axis=1)

    wrong_actual = tf.expand_dims(tf.concat(tf.unstack(
      tf.gather(self.sample_actual, wrong_images_idx), axis=0), axis=0), axis=0)
    
    wrong_recover = tf.expand_dims(tf.concat(tf.unstack(
      tf.gather_nd(
        tf.gather(sample_recover_each_y, wrong_images_idx),
        tf.stack([
         tf.range(nwrong, dtype=tf.int64),
         tf.gather(tf.argmax(self.y_recover, axis=-1), wrong_images_idx)], 
         axis=1)),
      axis=0), axis=0), axis=0)
    
    wrong_images = tf.concat(
      [wrong_bar_actual,
       wrong_actual, 
       wrong_recover,
       wrong_bar_recover], axis=2)
    wrong_images_summ = image_summary('hard_wrong_real_fake_imgs', wrong_images)
    summary_list += [wrong_images_summ]

    self.summary = merge_summary(summary_list)
    
    recover_path = os.path.join(
      self.checkpoint_dir,'recover_bs{}_epoch{}_lr{:.5g}'.format(
        config.recover_batch_size, config.recover_epoch, config.recover_learning_rate),
      datetime.now().strftime("%Y%m%d-%H%M%S"))
    print('Saving to recovery plots to {}.'.format(recover_path))
    try:
      os.mkdir(recover_path)
    except:
      pass
    self.writer = SummaryWriter(recover_path, self.sess.graph)

    recover_optim = tf.train.GradientDescentOptimizer(
        config.recover_learning_rate,
    ).minimize(self.mse_loss, var_list=[self.z_recover, 
                                        self.y_logit_recover,
                                        # self.y_recover,
                                       ])
    
    unintialized_variables = set(tf.all_variables()) - intialized_variables # https://stackoverflow.com/a/35618160
    self.sess.run(tf.variables_initializer(unintialized_variables))
     
    feed_dict = {}
    random_batch_idx = np.random.randint(
      len(self.data_X), size=[config.recover_batch_size])
    feed_dict[self.sample_actual] = self.data_X[random_batch_idx]
    feed_dict[self.y_actual] = self.data_y_actual[random_batch_idx]

    counter = 1
    start_time = time.time()

    for epoch in range(config.recover_epoch):
      _, summary_str, mse_loss, zero_one_loss = self.sess.run(
        [recover_optim, self.summary, self.mse_loss, self.zero_one_loss],
        feed_dict=feed_dict)
      self.writer.add_summary(summary_str, counter)

      counter += 1
      if (epoch+1)%100 == 0:
        print(
          "Recover Epoch: [%2d] time: %4.2f, mse_loss: %.5g, zeroone_loss: %.5g" 
          % (epoch, time.time()-start_time, mse_loss, zero_one_loss))



  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      if self.disc_type=="projection":     
        x = image 
        if self.config.concat_y and 1 in self.config.concat_y_layers: # proj with bottom concat
          yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
          x = conv_cond_concat(x, yb)
        h0 = lrelu(conv2d(
          x, self.df_dim, spectral_norm=self.config.spectral_norm,
          name='d_h0_conv'))

        if self.config.concat_y and 2 in self.config.concat_y_layers: # proj with 2nd concat
          yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
          h0 = conv_cond_concat(h0, yb)
        h1 = lrelu(self.d_bn1(conv2d(
          h0, self.df_dim, spectral_norm=self.config.spectral_norm,
          name='d_h1_conv')))

        if self.config.concat_y and 3 in self.config.concat_y_layers: # proj with 3rd concat
          yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
          h1 = conv_cond_concat(h1, yb)
        h2 = lrelu(self.d_bn2(conv2d(
          h1, self.df_dim, spectral_norm=self.config.spectral_norm,
          name='d_h2_conv')))

        if self.config.concat_y and 4 in self.config.concat_y_layers: # proj with 4th concat
          yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
          h2 = conv_cond_concat(h2, yb)
        h3 = lrelu(self.d_bn3(conv2d(
          h2, self.df_dim, spectral_norm=self.config.spectral_norm,
          name='d_h3_conv')))

        h3 = tf.reduce_mean(h3,axis=(1,2))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin',
                    max_norm=self.config.max_norm) 
        h5 = linear(tf.reshape(y, [self.batch_size, 10]),
                    self.df_dim, 'd_h5_y_lin', max_norm=self.config.max_norm)
        h6 = h4 + tf.reduce_sum(h3*h5, axis=1, keep_dims=True)
        # h6 = linear(lrelu(h6), 10, scope='d_h6_lin')
        return tf.nn.sigmoid(h6), h6
      else:     
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        x = conv_cond_concat(image, yb)

        h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
        h0 = conv_cond_concat(h0, yb)

        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
        h1 = tf.reshape(h1, [self.batch_size, -1])      
        h1 = concat([h1, y], 1)

        h3 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h3_lin')))
        h3 = concat([h3, y], 1)

        h4 = linear(h3, 1, 'd_h4_lin')

        return tf.nn.sigmoid(h4), h4

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      if True:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        h0 = tf.nn.relu(
            self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
            [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(
            deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def gen_sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()
      if True:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(
            deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def classifier(self, x, reuse=False):
    with tf.variable_scope("classifier") as scope:
      if reuse:
        scope.reuse_variables()
      # 1 layer NN
      hidden_layer = linear(tf.reshape(x, [self.batch_size, -1]), self.y_dim,
                            'd_classifier_h1')
      logits = hidden_layer

    return logits
    
  def load_mnist(self):
    data_dir = os.path.join(self.data_dir, self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_actual = np.zeros((len(y), self.y_dim), dtype=np.float)
    y_real = np.zeros((len(y), self.y_dim), dtype=np.float)
    y_fake = np.zeros((len(y), self.y_dim), dtype=np.float)
    y_gen = np.zeros((len(y), self.y_dim), dtype=np.float)
    y_real_weights = np.zeros((len(y), self.y_dim), dtype=np.float)
    
    # One-coin confusion matrix
    if not self.config.confusion_class_depend:
      C = ((1-self.alpha)/9.0)*np.ones((10,10)) + (self.alpha - (1-self.alpha)/9.0)*np.eye((10)) 
    #Random confusion matrix
    else:
      C = np.zeros((10,10))
      mean_diag = np.linspace(0.15, -0.15+2*self.alpha)
      for i in range(10):
        C[i,:] = (1.-mean_diag[i])/9.
        C[i,i] = mean_diag[i]
    self.confusion_matrix_actual = C    
    C_inv = np.linalg.inv(self.confusion_matrix_actual)
    print('C=\n', C)
    print('C_inv=\n', C_inv)
    for i, label in enumerate(y):
      y_actual[i, label] = 1
      y_real[i] = np.random.multinomial(1, C[y[i],:], size=1)
      y_real_weights[i] = C_inv[np.where(y_real[i]==1)[0],:]

      y_gen_label = np.random.randint(10, size=1)
      y_gen[i,int(y_gen_label)] = 1
      if self.config.real_match:
        y_gen[i] = y_real[i]
        y_gen_label = np.argmax(y_gen[i])

      y_fake[i] = np.random.multinomial(1, C[int(y_gen_label),:], size=1)  
    print(y_real[i], y_real_weights[i], end='')
    return X/255., y_actual, y_real, y_gen, y_fake, y_real_weights

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0