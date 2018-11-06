"""

"""

import numpy as np
import tensorflow as tf
import imageio
import os
import glob
import pickle

import common.misc

IMSIZE = 128
LOAD_SIZE = IMSIZE
INPUT_DIR = '/media/newhd/data/ILSVRC2012/train/ILSVRC2012_img_train'
OUTPUT_DIR = '/media/newhd/data/ILSVRC2012/train/resized_128'


def load_data(file_):
    """
      Load image names and corresponding labels.
    """
    data = np.genfromtxt(file_, dtype=str, comments=None, delimiter=' ')

    img_pathes = data[:, 0]
    img_labels = data[:, 1]

    return img_pathes, img_labels


def load_filenames():
    filepathes = glob.glob('*/*.JPEG')
    np.savetxt('../filenames.txt', filepathes, fmt="%s", comments=None)
    print('Load filenames: (%d)' % len(filepathes))  # 1281167
    return filepathes


def save_data_list(input_dir, filepathes):
    """
      Read, resize and save images listed in filepathes.
    """

    cnt = 0
    bad_img = list()
    for filepath in filepathes:
        image_path = os.path.join(input_dir, filepath)
        img, path = common.misc.get_image(image_path, LOAD_SIZE, is_crop=False)
        if img is None:
            bad_img.append(path)
            np.savetxt('../bad_img.txt', bad_img, fmt='%s', comments=None)
            continue
        img = img.astype('uint8')

        output_file = os.path.join(OUTPUT_DIR, filepath)
        if not os.path.exists(os.path.dirname(output_file)):
            os.mkdir(os.path.dirname(output_file))
        imageio.imwrite(output_file, img)

        cnt += 1
        if cnt % 1000 == 0:
            print('Resizing %d / %d' % (cnt, len(filepathes)))


def resize_ILSVRC2012_dataset(input_dir):
    # For train data
    # train_ = os.path.join(root_dir, 'caffe_ilsvrc12/train.txt')
    # train_filepathes, train_labels = load_data(train_)
    train_filepathes = load_filenames()
    save_data_list(input_dir, train_filepathes)

    # # For val data
    # val_ = os.path.join(input_dir, 'caffe_ilsvrc12/val.txt')
    # val_filepathes, test_labels = load_data(val_)
    # save_data_list(input_dir, val_filepathes)


def check_labels():
    """
      map_fn used in tf.data.Dataset
    """
    img_pathes, img_labels = load_data('../caffe_ilsvrc12/train.txt')
    img_labels = img_labels.astype(np.int32)
    print('len(img_pathes): {}'.format(len(img_pathes)))
    print('img_pathes[0]: {}'.format(img_pathes[0]))
    print('len(img_labels): {}'.format(len(img_labels)))
    print('img_labels[0]: {}'.format(img_labels[0]))
    img_dict = dict(zip(img_pathes, img_labels))

    img_glob = glob.glob('*/*.JPEG')
    train_data = dict()
    names_lost = list()
    for path in img_glob:
        if img_dict.get(path, -1) == -1:
            names_lost.append(path)
        else:
            train_data[path] = img_dict.get(path, -1)

    print('len(names_lost): {}'.format(len(names_lost)))
    np.savetxt('../names_lost.txt', names_lost, fmt='%s', comments=None)

    print('len(train_data): {}'.format(len(train_data)))
    with open('../train.pkl', 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)


# ###################  tf.data.Dataset  ################### #
def get_filenames_labels(root_dir):
    """
      Load image names and corresponding labels.
    """
    data = np.genfromtxt(os.path.join(root_dir, 'caffe_ilsvrc12/train.txt'), dtype=str, comments=None, delimiter=' ')
    img_pathes = data[:, 0]
    img_pathes = np.asarray(img_pathes).astype(np.str)
    img_labels = data[:, 1]
    img_labels = np.asarray(img_labels).astype(np.int32)
    filenames_labels = dict(zip(img_pathes, img_labels))

    filenames_ = np.genfromtxt(os.path.join(root_dir, 'filenames.txt'), dtype=str, comments=None, delimiter=' ')
    filenames_ = np.asarray(filenames_).astype(np.str)

    labels_ = [filenames_labels.get(filename) for filename in filenames_]
    labels_ = np.asarray(labels_).astype(np.int32)

    filenames_ = [os.path.join(root_dir, 'resized_128', filename) for filename in filenames_]
    filenames_ = np.asarray(filenames_).astype(np.str)

    shuffle_indices = np.random.permutation(np.arange(len(filenames_)))
    filenames_ = filenames_[shuffle_indices]
    labels_ = labels_[shuffle_indices]

    print('len(filenames_): {}'.format(len(filenames_)))
    print('len(labels_): {}'.format(len(labels_)))

    return filenames_, labels_


def _parse_function(filename, label):
    """
      map_fn used in tf.data.Dataset
    """
    image_string = tf.read_file(filename=filename)
    # image_decoded = tf.image.decode_jpeg(contents=image_string, channels=3)
    image_decoded = tf.image.decode_image(contents=image_string, channels=3)
    # image_decoded = tf.cast(image_decoded, tf.int32)
    # image_resized = tf.image.resize_images(images=image_decoded, size=[LOAD_SIZE, LOAD_SIZE])

    return image_decoded, label


def input_fn(filenames, labels, batch_size, num_epochs=1):
    """
      Store image and label in tfrecor format.

    Return:
    """
    filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    dataset = dataset.map(_parse_function)
    # dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.repeat(count=num_epochs)
    dataset = dataset.batch(batch_size=batch_size)

    iterator = dataset.make_one_shot_iterator()
    images, labels_ = iterator.get_next()

    return images, labels_


if __name__ == '__main__':
    resize_ILSVRC2012_dataset(INPUT_DIR)
