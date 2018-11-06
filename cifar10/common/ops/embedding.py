"""

"""

import numpy as np
import tensorflow as tf


# from libs.sn import spectral_normed_weight


def embed_y(inputs, vocab_size=1000, embedding_dim=300, word2vec_file=None,
            spectral_normed=False, update_collection=None, reuse=False):
    """
    Args:
      inputs: int, (batch size, 1).
      vocab_size: int, for cifar-10, it is 10.
      embedding_dim: int, embedding dim.
      word2vec_file:
      spectral_normed:
      update_collection:
      reuse:

    Returns:
        tensor of shape (batch size, embedding_dim)
    """
    # with tf.name_scope(name) as scope:
    with tf.variable_scope("Embedding.Label"):
        def uniform(size):
            return np.random.uniform(
                low=-0.08,
                high=0.08,
                size=size
            ).astype('float32')

        if word2vec_file is None:
            filter_values = uniform(
                (vocab_size, embedding_dim)
            )
            embedding_map = tf.get_variable(name='embedding_map',
                                            dtype=tf.float32,
                                            initializer=filter_values,
                                            trainable=True)
        else:
            filter_values = word2vec_file
            embedding_map = tf.get_variable(name='embedding_map',
                                            dtype=tf.float32,
                                            initializer=filter_values,
                                            trainable=False)

        return tf.nn.embedding_lookup(embedding_map, inputs)
