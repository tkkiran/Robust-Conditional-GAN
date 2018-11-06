# !/usr/bin/env python

# from mincepie import mapreducer, launcher
# import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import sys
import subprocess
import imageio
import errno
import scipy.misc
from scipy.misc import imsave


# from https://github.com/chainer/chainerrl/blob/f119a1fe210dd31ea123d244258d9b5edc21fba4/chainerrl/misc/copy_param.py
def record_setting(out):
    """Record scripts and commandline arguments"""
    out = out.split()[0].strip()
    if not os.path.exists(out):
        os.mkdir(out)
    subprocess.call("cp *.py %s" % out, shell=True)

    with open(out + "/command.txt", "w") as f:
        f.write(" ".join(sys.argv) + "\n")


# https://github.com/BVLC/caffe/blob/master/tools/extra/resize_and_crop_images.py
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_lib', 'opencv',
                           'OpenCV or PIL, case insensitive. The default value is the faster OpenCV.')
tf.app.flags.DEFINE_string('input_folder', '',
                           'The folder that contains all input images, organized in synsets.')
tf.app.flags.DEFINE_integer('output_side_length', 256,
                            'Expected side length of the output image.')
tf.app.flags.DEFINE_string('output_folder', '',
                           'The folder that we write output resized and cropped images to')


class OpenCVResizeCrop:
    def resize_and_crop_image(self, input_file, output_file, output_side_length=256):
        """Takes an image name, resize it and crop the center square
        """
        img = cv2.imread(input_file)
        height, width, depth = img.shape
        new_height = output_side_length
        new_width = output_side_length
        if height > width:
            new_height = output_side_length * height / width
        else:
            new_width = output_side_length * width / height
        resized_img = cv2.resize(img, (new_width, new_height))
        height_offset = (new_height - output_side_length) / 2
        width_offset = (new_width - output_side_length) / 2
        cropped_img = resized_img[height_offset:height_offset + output_side_length,
                      width_offset:width_offset + output_side_length]
        cv2.imwrite(output_file, cropped_img)


class PILResizeCrop:
    # http://united-coders.com/christian-harms/image-resizing-tips-every-coder-should-know/
    def resize_and_crop_image(self, input_file, output_file, output_side_length=256, fit=True):
        """Downsample the image.
        """
        img = Image.open(input_file)
        box = (output_side_length, output_side_length)
        # preresize image with factor 2, 4, 8 and fast algorithm
        factor = 1
        while img.size[0] / factor > 2 * box[0] and img.size[1] * 2 / factor > 2 * box[1]:
            factor *= 2
        if factor > 1:
            img.thumbnail((img.size[0] / factor, img.size[1] / factor), Image.NEAREST)

        # calculate the cropping box and get the cropped part
        if fit:
            x1 = y1 = 0
            x2, y2 = img.size
            wRatio = 1.0 * x2 / box[0]
            hRatio = 1.0 * y2 / box[1]
            if hRatio > wRatio:
                y1 = int(y2 / 2 - box[1] * wRatio / 2)
                y2 = int(y2 / 2 + box[1] * wRatio / 2)
            else:
                x1 = int(x2 / 2 - box[0] * hRatio / 2)
                x2 = int(x2 / 2 + box[0] * hRatio / 2)
            img = img.crop((x1, y1, x2, y2))

        # Resize the image with best quality algorithm ANTI-ALIAS
        img.thumbnail(box, Image.ANTIALIAS)

        # save it into a file-like object
        with open(output_file, 'wb') as out:
            img.save(out, 'JPEG', quality=75)


# class ResizeCropImagesMapper(mapreducer.BasicMapper):
#     '''The ImageNet Compute mapper.
#     The input value would be the file listing images' paths relative to input_folder.
#     '''
#
#     def map(self, key, value):
#         if type(value) is not str:
#             value = str(value)
#         files = [value]
#         image_lib = FLAGS.image_lib.lower()
#         if image_lib == 'pil':
#             resize_crop = PILResizeCrop()
#         else:
#             resize_crop = OpenCVResizeCrop()
#         for i, line in enumerate(files):
#             try:
#                 line = line.replace(FLAGS.input_folder, '').strip()
#                 line = line.split()
#                 image_file_name = line[0]
#                 input_file = os.path.join(FLAGS.input_folder, image_file_name)
#                 output_file = os.path.join(FLAGS.output_folder, image_file_name)
#                 output_dir = output_file[:output_file.rfind('/')]
#                 if not os.path.exists(output_dir):
#                     os.makedirs(output_dir)
#                 feat = resize_crop.resize_and_crop_image(input_file, output_file,
#                                                          FLAGS.output_side_length)
#             except Exception, e:
#                 # we ignore the exception (maybe the image is corrupted?)
#                 print(line, Exception, e)
#         yield value, FLAGS.output_folder


# mapreducer.REGISTER_DEFAULT_MAPPER(ResizeCropImagesMapper)
# mapreducer.REGISTER_DEFAULT_REDUCER(mapreducer.NoPassReducer)
# mapreducer.REGISTER_DEFAULT_READER(mapreducer.FileReader)
# mapreducer.REGISTER_DEFAULT_WRITER(mapreducer.FileWriter)


# ------


# Some codes from https://github.com/openai/improved-gan/blob/master/imagenet/utils.py
def get_image(image_path, image_size, is_crop=False, bbox=None):
    global index
    img, path = imread(image_path)
    if img is not None:
        out = transform(img, image_size, is_crop, bbox)
    else:
        out = None
    return out, path


def custom_crop(img, bbox):
    # bbox = [x-left, y-top, width, height]
    imsiz = img.shape  # [height, width, channel]
    # if box[0] + box[2] >= imsiz[1] or\
    #     box[1] + box[3] >= imsiz[0] or\
    #     box[0] <= 0 or\
    #     box[1] <= 0:
    #     box[0] = np.maximum(0, box[0])
    #     box[1] = np.maximum(0, box[1])
    #     box[2] = np.minimum(imsiz[1] - box[0] - 1, box[2])
    #     box[3] = np.minimum(imsiz[0] - box[1] - 1, box[3])
    center_x = int((2 * bbox[0] + bbox[2]) / 2)
    center_y = int((2 * bbox[1] + bbox[3]) / 2)
    R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
    y1 = np.maximum(0, center_y - R)
    y2 = np.minimum(imsiz[0], center_y + R)
    x1 = np.maximum(0, center_x - R)
    x2 = np.minimum(imsiz[1], center_x + R)
    img_cropped = img[y1:y2, x1:x2, :]
    return img_cropped


def transform(image, image_size, is_crop, bbox):
    image = colorize(image)
    if is_crop:
        image = custom_crop(image, bbox)
    #
    transformed_image = \
        scipy.misc.imresize(image, [image_size, image_size], 'bicubic')
    return np.array(transformed_image)


def imread(path):
    try:
        img = imageio.imread(path)
        img = img.astype(np.float)
    except Exception:
        img = None

    if img is None or img.shape == 0:
        # raise ValueError(path + " got loaded as a dimensionless array!")
        img = None
    return img, path


def colorize(img):
    if img.ndim == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
        img = np.concatenate([img, img, img], axis=2)
    if img.shape[2] == 4:
        img = img[:, :, 0:3]
    return img


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# Image grid saver, based on color_grid_vis from github.com/Newmu
def save_images(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.float):
        X = (255.99 * X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, int(n_samples / rows)

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        # X = X.transpose(0, 2, 3, 1)
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw))

    for n, x in enumerate(X):
        j = int(n / nw)
        i = int(n % nw)
        img[j * h:j * h + h, i * w:i * w + w] = x

    imsave(save_path, img)


def get_z(batchsize, n_hidden=128):
    """Get random noise 'z'.

    Args:
      batchsize:
      n_hidden:

    Returns:
    """
    z = np.random.normal(size=(batchsize, n_hidden)).astype(np.float32)
    # z /= np.sqrt(np.sum(z * z, axis=1, keepdims=True) / n_hidden + 1e-8)
    return z


# ------


def scope_has_variables(scope):
    """

    Args:
      scope:

    Returns:
    """
    return len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)) > 0


def optimistic_restore(session, save_file):
    """

    Args:
      session:
      save_file:

    Returns:
    """
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []

    name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))

    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

    # print('\n--------variables stored:--------')
    # for var_name, saved_var_name in var_names:
    #     print(var_name)

    print('\n--------variables to restore:--------')
    for var in restore_vars:
        print(var)


def get_loss(disc_real, disc_fake, loss_type='HINGE'):
    """
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/losses/python/losses_impl.py

    Args:
      disc_real:
      disc_fake:
      loss_type:

    Returns:
    """
    if loss_type == 'HINGE':
        disc_real_l = tf.reduce_mean(tf.nn.relu(1.0 - disc_real))
        disc_fake_l = tf.reduce_mean(tf.nn.relu(1.0 + disc_fake))
        d_loss = disc_real_l + disc_fake_l

        g_loss = -tf.reduce_mean(disc_fake)
    elif loss_type == 'WGAN':
        disc_real_l = - tf.reduce_mean(disc_real)
        disc_fake_l = tf.reduce_mean(disc_fake)
        d_loss = disc_real_l + disc_fake_l

        # clip_d_vars_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for var in d_vars]
        # # Paste the code bellow to where `session.run(d_train_op)`
        # session.run(clip_d_vars_op)

        g_loss = -tf.reduce_mean(disc_fake)
    elif loss_type == 'WGAN-GP':
        disc_real_l = - tf.reduce_mean(disc_real)
        disc_fake_l = tf.reduce_mean(disc_fake)
        d_loss = disc_real_l + disc_fake_l

        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/losses/python/losses_impl.py#L301
        # Paste the code bellow where `get_loss()` is called.
        # # Gradient Penalty
        # alpha = tf.random_uniform(shape=[args.batch_size, 1, 1, 1], minval=0., maxval=1.)
        # differences = x_fake - real_data
        # interpolates = real_data + (alpha * differences)
        # gradients = tf.gradients(
        #     model.get_discriminator(interpolates, real_labels, 'NO_OPS', reuse=True)[0], [interpolates])[0]
        # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]) + 1e-10)
        # gradient_penalty = 10 * tf.reduce_mean(tf.square((slopes - 1.)))
        # d_loss_gan += gradient_penalty

        g_loss = -tf.reduce_mean(disc_fake)
    elif loss_type == 'LSGAN':
        # L = 1/2 * (D(x) - `real`) ** 2 + 1/2 * (D(G(z)) - `fake_label`) ** 2
        disc_real_l = tf.reduce_mean(tf.square(1.0 - disc_real))
        disc_fake_l = tf.reduce_mean(tf.square(disc_fake))
        d_loss = (disc_real_l + disc_fake_l) / 2.0

        # L = 1/2 * (D(G(z)) - `real_label`) ** 2
        g_loss = tf.reduce_mean(tf.square(1.0 - disc_fake)) / 2.0
    elif loss_type == 'CGAN':
        disc_real_l = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                    labels=tf.ones_like(disc_real)))
        disc_fake_l = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                    labels=tf.zeros_like(disc_fake)))
        d_loss = disc_real_l + disc_fake_l

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                    labels=tf.ones_like(disc_fake)))
    elif loss_type == 'Modified_MiniMax':
        # L = - real_weights * log(sigmoid(D(x)))
        #     - generated_weights * log(1 - sigmoid(D(G(z))))
        disc_real_l = -tf.reduce_mean(tf.log(tf.nn.sigmoid(disc_real)))
        disc_fake_l = -tf.reduce_mean(tf.log(1.0 - tf.nn.sigmoid(disc_fake)))
        d_loss = disc_real_l + disc_fake_l

        # L = -log(sigmoid(D(G(z))))
        g_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(disc_fake)))
    elif loss_type == 'MiniMax':
        # L = - real_weights * log(sigmoid(D(x)))
        #     - generated_weights * log(1 - sigmoid(D(G(z))))
        disc_real_l = -tf.reduce_mean(tf.log(tf.nn.sigmoid(disc_real)))
        disc_fake_l = -tf.reduce_mean(tf.log(1.0 - tf.nn.sigmoid(disc_fake)))
        d_loss = disc_real_l + disc_fake_l

        # L = log(sigmoid(D(x))) + log(1 - sigmoid(D(G(z))))
        g_loss = tf.reduce_mean(tf.log(1.0 - tf.nn.sigmoid(disc_fake)))

    return d_loss, g_loss


if __name__ == '__main__':
    tf.app.run()
