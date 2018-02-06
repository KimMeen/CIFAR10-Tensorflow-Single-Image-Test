# -*- coding: utf-8 -*-
"""
@author: Ming JIN
"""

import tensorflow as tf
import numpy
#import cv2
from PIL import Image
from PIL import ImageFilter

image_path = "./test_images/ship.jpg"  #指定测试图片
input_number = 8                      #指定测试图片编号（0-9）

image_size = 24
labels = ['飞机','汽车','鸟','猫','麋鹿','狗','青蛙','马','轮船','卡车',]

def _image_test_crop(images, crop_shape=(24,24,3)):
        # 图像切割
        new_images = numpy.empty((images.shape[0],24,24,3))
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            left = int((old_image.shape[0] - crop_shape[0])/2)
            top = int((old_image.shape[1] - crop_shape[1])/2)
            new_image = old_image[left:left+crop_shape[0],top:top+crop_shape[1], :]
            new_images[i,:,:,:] = new_image
            
        return new_images

def _image_whitening(images):
        # 图像白化
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            new_image = (old_image - numpy.mean(old_image)) / numpy.std(old_image)
            images[i,:,:,:] = new_image
        
        return images

def conv_weight_variable(kernal_shape,input_shape):
    initial = tf.truncated_normal(kernal_shape, stddev=numpy.sqrt(2.0 / (input_shape[0] * input_shape[1] * input_shape[2])))
    return tf.Variable(initial)

def fc_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=numpy.sqrt(2.0 / shape[0]))
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv_batch_normal(x,b,n_filter,is_train=-1):
    epsilon = 1e-5
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n_filter]),trainable=False)
    batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
  
    def mean_var_with_update():      
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, variance = tf.cond(tf.greater(is_train,0), mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))

    return tf.nn.batch_normalization(x, mean, variance, b, gamma, epsilon)

def fc_batch_normal(x,b,hidden_dim,is_train=-1):
    epsilon = 1e-5
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[hidden_dim]),trainable=False)
    batch_mean, batch_var = tf.nn.moments(x, axes=[0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
  
    def mean_var_with_update():      
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, variance = tf.cond(tf.greater(is_train,0), mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))

    return tf.nn.batch_normalization(x, mean, variance, b, gamma, epsilon)  

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
x_image = tf.reshape(xs, [-1, 24, 24, 3])

## conv1 layer ##
W_conv1 = conv_weight_variable([3,3,3,64],[image_size,image_size,3]) 
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv_batch_normal(conv2d(x_image, W_conv1),b_conv1,64)) 
h_pool1 = max_pool_2x2(h_conv1)                          
norm1 = tf.nn.local_response_normalization(h_pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

## conv2 layer ##
W_conv2 = conv_weight_variable([3,3,64,128],[int(image_size/2),int(image_size/2),64]) 
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv_batch_normal(conv2d(norm1, W_conv2),b_conv2,128))
h_pool2 = max_pool_2x2(h_conv2)                          
norm2 = tf.nn.local_response_normalization(h_pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

## conv3 layer ##
W_conv3 = conv_weight_variable([3,3,128,256],[int(image_size/4),int(image_size/4),128]) 
b_conv3 = bias_variable([256])
h_conv3 = tf.nn.relu(conv_batch_normal(conv2d(norm2, W_conv3),b_conv3,256))
h_pool3 = max_pool_2x2(h_conv3)                          
norm3 = tf.nn.local_response_normalization(h_pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

## fc1 layer ##
W_fc1 = fc_weight_variable([int(image_size/8)*int(image_size/8)*256, 1024])
b_fc1 = bias_variable([1024])
h_norm3_flat = tf.reshape(norm3, [-1,int(image_size/8)*int(image_size/8)*256])
intermediate_fc1 = fc_batch_normal(tf.matmul(h_norm3_flat, W_fc1),b_fc1,1024)
h_fc1 = tf.nn.relu(intermediate_fc1)

## fc2 layer ##
W_fc2 = fc_weight_variable([1024, 10])
b_fc2 = bias_variable([10])
intermediate_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
h_fc2 = intermediate_fc2

## softmax logic layer ##
prediction = tf.nn.softmax(h_fc2)

#############################################
image = Image.open(image_path)

if (image.size != (32,32)):
    image = image.resize((32,32),Image.ANTIALIAS)
    image = image.filter(ImageFilter.DETAIL)

#image.save('./test_images/cat3_0_0.jpg')
image = numpy.reshape(image, [1, 32, 32, 3])
image = numpy.multiply(image, 1.0 / 255.0)
image = _image_test_crop(image)
image = _image_whitening(image)

#############################################

with tf.Session() as sess:
   saver = tf.train.Saver()
   ckpt = tf.train.get_checkpoint_state('./model/')
   saver.restore(sess, ckpt.model_checkpoint_path)
   print("\nModel has been restored.\n")
   
   result = sess.run(prediction, feed_dict={xs:image})
   
   max_index = numpy.argmax(result)
   print("The input labels is :",labels[input_number])
   print("The outpur labels is :",labels[max_index])
   
   if (labels[input_number] == labels[max_index]):
       print("预测结果正确")
   else:
       print("预测结果错误")       
   
   print("置信率为:",result[:,max_index])

sess.close()