"""
Modification of https://github.com/carpedm20/DCGAN-tensorflow/blob/master/download.py
Modification of https://github.com/stanfordnlp/treelstm/blob/master/scripts/download.py

Downloads the following:
- MNIST dataset
- CIFAR10 dataset
"""

from __future__ import print_function
import os
import argparse
import subprocess


parser = argparse.ArgumentParser(description='Download dataset for RCGAN.')
parser.add_argument('datasets', metavar='N', type=str, nargs='+', choices=['cifar10', 'mnist'],
                    help='name of dataset to download [cifar10, mnist]')

def download_mnist(dirpath):
  data_dir = os.path.join(dirpath, 'mnist')
  if os.path.exists(data_dir):
    print('Found MNIST - skip')
    return
  else:
    os.mkdir(data_dir)
  url_base = 'http://yann.lecun.com/exdb/mnist/'
  file_names = ['train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz']
  for file_name in file_names:
    url = (url_base+file_name).format(**locals())
    print(url)
    out_path = os.path.join(data_dir, file_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading ', file_name)
    subprocess.call(cmd)
    cmd = ['gzip', '-d', out_path]
    print('Decompressing ', file_name)
    subprocess.call(cmd)

def download_cifar10(dirpath):
  data_dir = os.path.join(dirpath, 'cifar10')
  if not os.path.exists(data_dir):
    os.mkdir(data_dir)
  url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  print(url)

  out_path = os.path.join(data_dir, 'cifar-10-python.tar.gz')
  if os.path.join(out_path):
    print('{} already exists'.format(out_path))
  else:
    cmd = ['curl', url, '-o', out_path]
    print('Downloading ', url)
    print(' '.join(cmd))
    subprocess.call(cmd)
  cmd = ['tar', '-xzf', out_path, '-C', data_dir]
  print('Decompressing ', out_path)
  print(' '.join(cmd))
  subprocess.call(cmd)

def prepare_data_dir(path = './data'):
  if not os.path.exists(path):
    os.mkdir(path)

if __name__ == '__main__':
  args = parser.parse_args()
  prepare_data_dir()

  if any(name in args.datasets for name in ['cifar', 'cifar10', 'cifar-10']):
    download_cifar10('./data')
  elif 'mnist' in args.datasets:
    download_mnist('./data')
