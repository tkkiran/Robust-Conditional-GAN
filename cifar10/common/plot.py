import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import logging
import pickle
# import time
import os

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]


def tick():
    _iter[0] += 1


def plot(name, value):
    _since_last_flush[name][_iter[0]] = value


def flush():
    prints = []

    for name, vals in _since_last_flush.items():
        prints.append("{}: {}".format(name, np.mean(list(vals.values()))))
        _since_beginning[name].update(vals)

        x_vals = np.sort(list(_since_beginning[name].keys()))
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.savefig(name.replace(' ', '_') + '.jpg')

    # print("iter {}\n{}".format(_iter[0], ", ".join(prints)))
    logging.info("iter {}\n{}".format(_iter[0], ", ".join(prints)))
    _since_last_flush.clear()

    logging.info('starting dumping values into log.pkl')
    with open('log.pkl', 'wb') as f:
        pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)
    logging.info('done dumping values into log.pkl')



def dir_flush(dir, log_pkl=False):
    prints = []

    for name, vals in _since_last_flush.items():
        prints.append("{}: {}".format(name, np.mean(list(vals.values()))))
        _since_beginning[name].update(vals)

        x_vals = np.sort(list(_since_beginning[name].keys()))
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.savefig(os.path.join(dir, '{}.jpg'.format(name.replace(' ', '_')) ))

    # print("iter {}\n{}".format(_iter[0], ", ".join(prints)))
    logging.info("iter {}\n{}".format(_iter[0], ", ".join(prints)))
    _since_last_flush.clear()

    if log_pkl:
        logging.info('starting dumping values into log.pkl')
        with open(os.path.join(dir, 'log.pkl'), 'wb') as f:
            pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)
        logging.info('done dumping values into log.pkl')
