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
import cv2
import random

def read_data_sets(data_dir,
                   one_hot=True,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=0,                               #no validation needed
                   seed=None):

    TRAIN = os.path.join(data_dir, "train", "train.txt")
    TEST = os.path.join(data_dir, "test", "test.txt")
    # from tensorflow.examples.tutorials.mnist import input_data
    # train and test from images and txt labels
    train_images, train_labels = process_images(TRAIN, one_hot=one_hot)
    test_images, test_labels = process_images(TEST, one_hot=one_hot)

    #if not 0 <= validation_size <= len(train_images):
    #    raise ValueError(
    #        'Validation size should be between 0 and {}. Received: {}.'
    #            .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]
    
    cut = numpy.empty((128,24,24,3))
    cut_test = numpy.empty((1280,24,24,3))
    
    new_part_cut = numpy.empty((128,24,24,3))
    new_part_cut_test = numpy.empty((1280,24,24,3))
    rest_part_cut = numpy.empty((128,24,24,3))
    rest_part_cut_test = numpy.empty((1280,24,24,3))
    
    train = DataSet(train_images, train_labels, cut, cut_test, new_part_cut, rest_part_cut, new_part_cut_test, rest_part_cut_test, dtype=dtype, reshape=reshape, seed=seed)
    validation = DataSet(validation_images,validation_labels,cut,cut_test, new_part_cut, rest_part_cut, new_part_cut_test, rest_part_cut_test, dtype=dtype,reshape=reshape,seed=seed)
    test = DataSet(test_images, test_labels, cut, cut_test, new_part_cut, rest_part_cut, new_part_cut_test, rest_part_cut_test, dtype=dtype, reshape=reshape, seed=seed)

    return base.Datasets(train=train, validation=validation, test=test)


def process_images(label_file, one_hot, num_classes=10):
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
      return images.reshape(num_images, rows, cols, 3), dense_to_one_hot(numpy.array(labels, dtype=numpy.uint8), num_classes) #use this one
  
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
               cut_images,
               cut_test_images,
               images_new_part_cut,
               images_rest_part_cut,
               images_new_part_cut_test,
               images_rest_part_cut_test,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=False,
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
        
      assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 3   #原来是1，改成了3
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2], images.shape[3]) #加乘了一个images.shape[3]
      
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    
    self._cut_images = cut_images
    self._cut_test_images = cut_test_images
    self._images_new_part_cut = images_new_part_cut
    self._images_rest_part_cut = images_rest_part_cut
    self._images_new_part_cut_test = images_new_part_cut_test
    self._images_rest_part_cut_test = images_rest_part_cut_test   
    
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

  def next_batch(self, batch_size, shuffle, flip, whiten, noise, crop, crop_test):
    """Return the next `batch_size` examples from this data set."""
    #if fake_data:
    #  fake_image = [1] * 3072  #原来是1024，改成3072
    #  if self.one_hot:
    #    fake_label = [1] + [0] * 9
    #  else:
    #    fake_label = 0
    #  return [fake_image for _ in range(batch_size)], [
    #      fake_label for _ in range(batch_size)
    #  ]
        
    start = self._index_in_epoch #上一次batch个图片的最后一张图片下边，从这里开始
    
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    
    # Go to the next epoch
    if start + batch_size > self._num_examples:  #本次epoch从 _index_in_epoch 到 _index_in_epoch + batch_size
                                                #如果这个条件满足了就说明了已经完成了对图片库一遍的遍历，需要重新洗牌再组合batch
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
 
      if crop:
         images_crop_1 = images_new_part
         #self._images[start:end] = self._image_crop(images_crop)
         self._images_new_part_cut = self._image_crop(images_crop_1)
         
         images_crop_2 = images_rest_part
         #self._images[start:end] = self._image_crop(images_crop)
         self._images_rest_part_cut = self._image_crop(images_crop_2)
         
      if crop_test:
         images_crop_1 = images_new_part
         #self._images[start:end] = self._image_crop(images_crop)
         self._images_new_part_cut_test = self._image_test_crop(images_crop_1)
         
         images_crop_2 = images_rest_part
         #self._images[start:end] = self._image_crop(images_crop)
         self._images_rest_part_cut_test = self._image_test_crop(images_crop_2) 
         
      if flip:
          images_flip_1 = self._images_new_part_cut
          self._images_new_part_cut = self._image_flip(images_flip_1)
          images_flip_2 = self._images_rest_part_cut
          self._images_rest_part_cut = self._image_flip(images_flip_2)
      
      if whiten:
          if crop:
             images_whiten_1 = self._images_new_part_cut
             self._images_new_part_cut = self._image_whitening(images_whiten_1)
             images_whiten_2 = self._images_rest_part_cut
             self._images_rest_part_cut = self._image_whitening(images_whiten_2)
             
          if crop_test:
             images_whiten_1 = self._images_new_part_cut_test
             self._images_new_part_cut_test = self._image_whitening(images_whiten_1)
             images_whiten_2 = self._images_rest_part_cut_test
             self._images_rest_part_cut_test = self._image_whitening(images_whiten_2)     
      
      if noise:
          images_noise_1 = self._images_new_part_cut
          self._images_new_part_cut = self._image_noise(images_noise_1)
          images_noise_2 = self._images_rest_part_cut
          self._images_rest_part_cut = self._image_noise(images_noise_2)
          
      if crop:
         return numpy.concatenate((self._images_rest_part_cut, self._images_new_part_cut), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
      elif crop_test:   
         return numpy.concatenate((self._images_rest_part_cut_test, self._images_new_part_cut_test), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      
      if crop:
         images_crop = self._images[start:end]
         #self._images[start:end] = self._image_crop(images_crop)
         self._cut_images = self._image_crop(images_crop)

      if crop_test:
         images_crop = self._images[start:end]
         #self._images[start:end] = self._image_crop(images_crop)
         self._cut_test_images = self._image_test_crop(images_crop)
          
      if flip:
          images_flip = self._cut_images
          self._cut_images = self._image_flip(images_flip)
      
      if whiten:
          if crop:
             images_whiten = self._cut_images
             self._cut_images = self._image_whitening(images_whiten)
          if crop_test:
             images_whiten = self._cut_test_images
             self._cut_test_images = self._image_whitening(images_whiten)     
      
      if noise:
          images_noise = self._cut_images
          self._cut_images = self._image_noise(images_noise)
      
      if crop:
         return self._cut_images, self._labels[start:end]
      elif crop_test:
         return self._cut_test_images, self._labels[start:end]     

########################数据增强相关函数#############################
  def _image_crop(self, images, crop_shape=(24,24,3)):
        # 图像切割
        new_images = numpy.empty((images.shape[0],24,24,3))
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            left = numpy.random.randint(old_image.shape[0] - crop_shape[0] + 1)
            top = numpy.random.randint(old_image.shape[1] - crop_shape[1] + 1)
            new_image = old_image[left:left+crop_shape[0],top:top+crop_shape[1], :]
            new_images[i,:,:,:] = new_image
            
        return new_images

  def _image_test_crop(self, images, crop_shape=(24,24,3)):
        # 图像切割
        new_images = numpy.empty((images.shape[0],24,24,3))
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            left = int((old_image.shape[0] - crop_shape[0])/2)
            top = int((old_image.shape[1] - crop_shape[1])/2)
            new_image = old_image[left:left+crop_shape[0],top:top+crop_shape[1], :]
            new_images[i,:,:,:] = new_image
            
        return new_images

  def _image_whitening(self, images):
        # 图像白化
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            new_image = (old_image - numpy.mean(old_image)) / numpy.std(old_image)
            images[i,:,:,:] = new_image
        
        return images
  
  def _image_flip(self, images):
        # 图像翻转
        for i in range(images.shape[0]):
            
            old_image = images[i,:,:,:]
            
            if numpy.random.random() < 0.5:
                #new_image_mid = numpy.fliplr(old_image.reshape(32,32,3))
                new_image = cv2.flip(old_image, 1)
            else:
                new_image = old_image
                
            images[i,:,:,:] = new_image
        
        return images

  def _image_noise(self, images, mean=0, std=0.01):
        # 图像噪声
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            new_image = old_image
            for i in range(old_image.shape[0]):
                for j in range(old_image.shape[1]):
                    for k in range(old_image.shape[2]):
                        new_image[i, j, k] += random.gauss(mean, std)
            images[i,:,:,:] = new_image
        
        return images