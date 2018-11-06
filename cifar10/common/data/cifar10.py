import numpy as np

import logging
import os
import urllib
import gzip
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        # print('-------------dict.keys()---------------')
        # print(dict.keys())
        # print('-------------dict.keys()---------------')
    return dict[b'data'], dict[b'labels']


def cifar_generator(filenames, batch_size, data_dir, C_ALPHA):
    all_data = []
    all_labels = []
    for filename in filenames:
        data, labels = unpickle(os.path.join(data_dir, filename))
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    labels_random = np.random.randint(10,size=50000)

    labels_biased = np.zeros((50000,))
    labels_inv_weights = np.zeros((50000,10))
    C_ALPHA_inv = np.linalg.inv(C_ALPHA)
    logging.debug('first 10 labels before noise: {}'.format(labels[0:10]))
    for i in range(len(labels)):
        labels[i] = np.nonzero(np.random.multinomial(1, C_ALPHA[labels[i],:], size=1))[1]
        labels_inv_weights[i] = C_ALPHA_inv[labels[i],:]        
        labels_biased[i] = np.nonzero(np.random.multinomial(1, C_ALPHA[labels_random[i],:], size=1))[1]
    logging.debug('first 10 labels after noise: {}'.format(labels[0:10]))   
        
    def get_epoch():
        for i in range(int(len(images) / batch_size)):
            yield (images[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size], labels_random[i * batch_size:(i + 1) * batch_size], labels_biased[i * batch_size:(i + 1) * batch_size], labels_inv_weights[i * batch_size:(i + 1) * batch_size])

    return get_epoch


def load(batch_size, data_dir, C_ALPHA):
    return (
        cifar_generator(['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'], batch_size, data_dir, C_ALPHA),
        cifar_generator(['test_batch'], batch_size, data_dir, C_ALPHA)
    )
