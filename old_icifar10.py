from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from PIL import Image
import fileUtil as file
from labelFile2Map import *
import base

def read_data_sets(data_dir,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None):

    TRAIN = os.path.join(data_dir, "train", "train.txt")
    TEST = os.path.join(data_dir, "test", "test.txt")
    # from tensorflow.examples.tutorials.mnist import input_data
    # train and test from images and txt labels
    train_images, train_labels = process_images(TRAIN, one_hot=one_hot)
    test_images, test_labels = process_images(TEST, one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
                .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(
        train_images, train_labels, dtype=dtype, reshape=reshape, seed=seed)
    validation = DataSet(
        validation_images,
        validation_labels,
        dtype=dtype,
        reshape=reshape,
        seed=seed)
    test = DataSet(
        test_images, test_labels, dtype=dtype, reshape=reshape, seed=seed)

    return base.Datasets(train=train, validation=validation, test=test)


def process_images(label_file, one_hot=False, num_classes=10):
    if file.getFileName(label_file) == 'train.txt':
        images = numpy.empty((50000, 3072)) #原来是1024，改成3072
        labels = numpy.empty(50000)
    if file.getFileName(label_file) == 'test.txt':
        images = numpy.empty((10000, 3072))
        labels = numpy.empty(10000)
    lines = readLines(label_file)
    label_record = map(lines)
    file_name_length = len(file.getFileName(label_file))
    image_dir = label_file[:-1*file_name_length]
    print(len(label_record))
    index = 0
    for name in label_record:
        # print label_record[name]
        image = Image.open(image_dir + str(label_record[name]) + '/' + name)
        if index % 100 == 0:
            print("processing %d: " % index + image_dir + str(label_record[name]) + '/' + name)

        img_ndarray = numpy.asarray(image, dtype='float32')
        images[index] = numpy.ndarray.flatten(img_ndarray)
        labels[index] = numpy.int(label_record[name])

        index = index + 1
    print("done: %d" % index)
    num_images = index
    rows = 32
    cols = 32
    if one_hot:
      return images.reshape(num_images, rows, cols, 3), dense_to_one_hot(numpy.array(labels, dtype=numpy.uint8), num_classes)
    return images.reshape(num_images, rows, cols, 3), numpy.array(labels, dtype=numpy.uint8)


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


class DataSet(object):
  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 3   #原来是1，改成了3
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2] * images.shape[3]) #加乘了一个images.shape[3]
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 3072  #原来是1024，改成3072
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in range(batch_size)], [
          fake_label for _ in range(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]
