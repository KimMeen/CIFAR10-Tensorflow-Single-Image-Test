#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/4/30 上午10:57
# @Author  : mrlittlepig
# @Site    : www.mrlittlepig.xyz
# @File    : base.py
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
from os import path
import random
import tempfile
import time

import numpy as np

from tensorflow.contrib.framework import deprecated
from tensorflow.python.platform import gfile

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def load_csv_with_header(filename,
                         target_dtype,
                         features_dtype,
                         target_column=-1):
  """Load dataset from CSV file with a header row."""
  with gfile.Open(filename) as csv_file:
    data_file = csv.reader(csv_file)
    header = next(data_file)
    n_samples = int(header[0])
    n_features = int(header[1])
    data = np.zeros((n_samples, n_features), dtype=features_dtype)
    target = np.zeros((n_samples,), dtype=target_dtype)
    for i, row in enumerate(data_file):
      target[i] = np.asarray(row.pop(target_column), dtype=target_dtype)
      data[i] = np.asarray(row, dtype=features_dtype)

  return Dataset(data=data, target=target)


def load_csv_without_header(filename,
                            target_dtype,
                            features_dtype,
                            target_column=-1):
  """Load dataset from CSV file without a header row."""
  with gfile.Open(filename) as csv_file:
    data_file = csv.reader(csv_file)
    data, target = [], []
    for row in data_file:
      target.append(row.pop(target_column))
      data.append(np.asarray(row, dtype=features_dtype))

  target = np.array(target, dtype=target_dtype)
  data = np.array(data)
  return Dataset(data=data, target=target)


def shrink_csv(filename, ratio):
  """Create a smaller dataset of only 1/ratio of original data."""
  filename_small = filename.replace('.', '_small.')
  with gfile.Open(filename_small, 'w') as csv_file_small:
    writer = csv.writer(csv_file_small)
    with gfile.Open(filename) as csv_file:
      reader = csv.reader(csv_file)
      i = 0
      for row in reader:
        if i % ratio == 0:
          writer.writerow(row)
        i += 1


def load_iris(data_path=None):
  """Load Iris dataset.

  Args:
      data_path: string, path to iris dataset (optional)

  Returns:
    Dataset object containing data in-memory.
  """
  if data_path is None:
    module_path = path.dirname(__file__)
    data_path = path.join(module_path, 'data', 'iris.csv')
  return load_csv_with_header(
      data_path,
      target_dtype=np.int,
      features_dtype=np.float)


def load_boston(data_path=None):
  """Load Boston housing dataset.

  Args:
      data_path: string, path to boston dataset (optional)

  Returns:
    Dataset object containing data in-memory.
  """
  if data_path is None:
    module_path = path.dirname(__file__)
    data_path = path.join(module_path, 'data', 'boston_house_prices.csv')
  return load_csv_with_header(
      data_path,
      target_dtype=np.float,
      features_dtype=np.float)


def retry(initial_delay,
          max_delay,
          factor=2.0,
          jitter=0.25,
          is_retriable=None):
  """Simple decorator for wrapping retriable functions.

  Args:
    initial_delay: the initial delay.
    factor: each subsequent retry, the delay is multiplied by this value.
        (must be >= 1).
    jitter: to avoid lockstep, the returned delay is multiplied by a random
        number between (1-jitter) and (1+jitter). To add a 20% jitter, set
        jitter = 0.2. Must be < 1.
    max_delay: the maximum delay allowed (actual max is
        max_delay * (1 + jitter).
    is_retriable: (optional) a function that takes an Exception as an argument
        and returns true if retry should be applied.
  """
  if factor < 1:
    raise ValueError('factor must be >= 1; was %f' % (factor,))

  if jitter >= 1:
    raise ValueError('jitter must be < 1; was %f' % (jitter,))

  # Generator to compute the individual delays
  def delays():
    delay = initial_delay
    while delay <= max_delay:
      yield delay * random.uniform(1 - jitter,  1 + jitter)
      delay *= factor

  def wrap(fn):
    """Wrapper function factory invoked by decorator magic."""

    def wrapped_fn(*args, **kwargs):
      """The actual wrapper function that applies the retry logic."""
      for delay in delays():
        try:
          return fn(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except)
          if is_retriable is None:
            continue

          if is_retriable(e):
            time.sleep(delay)
          else:
            raise
      return fn(*args, **kwargs)
    return wrapped_fn
  return wrap


_RETRIABLE_ERRNOS = {
    110,  # Connection timed out [socket.py]
}


def _is_retriable(e):
  return isinstance(e, IOError) and e.errno in _RETRIABLE_ERRNOS

def load_images(work_directory):
  filepath = os.path.join(work_directory)
  return filepath

def maybe_download(filename, work_directory):
  """Download the data from source url, unless it's already here.

  Args:
      filename: string, name of the file in the directory.
      work_directory: string, path to working directory.
      source_url: url to download from if file doesn't exist.

  Returns:
      Path to resulting file.
  """
  if not gfile.Exists(work_directory):
    gfile.MakeDirs(work_directory)
  filepath = os.path.join(work_directory, filename)
  return filepath
