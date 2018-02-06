# -*- coding: utf-8 -*-
"""
@author: Ming JIN
"""

import tensorflow as tf
import icifar10
import numpy
import time
import matplotlib.pyplot as plt

cifar10 = icifar10.read_data_sets('./cifar-10/')
iterration = 15001
batch_size = 128
lr1 = 0.1
lr2 = 0.01
lr3 = 0.001
lr4 = 0.0001
image_size = 24

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def compute_loss(v_xs, v_ys):
    avg_loss_value = sess.run(avg_loss, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 0.5})
    return avg_loss_value

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

def conv_batch_normal(x,b,n_filter,is_train=1):
    epsilon = 1e-5
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n_filter]),trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
  
    def mean_var_with_update():      
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, variance = tf.cond(tf.greater(is_train,0), mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))

    return tf.nn.batch_normalization(x, mean, variance, b, gamma, epsilon)

def fc_batch_normal(x,b,hidden_dim,is_train=1):
    epsilon = 1e-5
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[hidden_dim]),trainable=True)
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
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)


## conv1 layer ##
W_conv1 = conv_weight_variable([3,3,3,64],[image_size,image_size,3])
weight_decay1 = tf.multiply(tf.nn.l2_loss(W_conv1), 1e-4)
tf.add_to_collection('losses', weight_decay1)
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv_batch_normal(conv2d(xs, W_conv1),b_conv1,64)) 
h_pool1 = max_pool_2x2(h_conv1)                         
norm1 = tf.nn.local_response_normalization(h_pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

## conv2 layer ##
W_conv2 = conv_weight_variable([3,3,64,128],[int(image_size/2),int(image_size/2),64]) 
weight_decay2 = tf.multiply(tf.nn.l2_loss(W_conv2), 1e-4)
tf.add_to_collection('losses', weight_decay2)
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv_batch_normal(conv2d(norm1, W_conv2),b_conv2,128))
h_pool2 = max_pool_2x2(h_conv2)
norm2 = tf.nn.local_response_normalization(h_pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

## conv3 layer ##
W_conv3 = conv_weight_variable([3,3,128,256],[int(image_size/4),int(image_size/4),128]) 
weight_decay3 = tf.multiply(tf.nn.l2_loss(W_conv3), 1e-4)
tf.add_to_collection('losses', weight_decay3)
b_conv3 = bias_variable([256])
h_conv3 = tf.nn.relu(conv_batch_normal(conv2d(norm2, W_conv3),b_conv3,256))
h_pool3 = max_pool_2x2(h_conv3)                        
norm3 = tf.nn.local_response_normalization(h_pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

## fc1 layer ##
W_fc1 = fc_weight_variable([int(image_size/8)*int(image_size/8)*256, 1024])
b_fc1 = bias_variable([1024])
h_norm3_flat = tf.reshape(norm3, [-1,int(image_size/8)*int(image_size/8)*256])
intermediate_fc1 = fc_batch_normal(tf.matmul(h_norm3_flat, W_fc1),b_fc1,1024)
h_fc1_drop = tf.nn.dropout(intermediate_fc1, keep_prob)
h_fc1 = tf.nn.relu(h_fc1_drop)

## fc2 layer ##
W_fc2 = fc_weight_variable([1024, 10])
b_fc2 = bias_variable([10])
intermediate_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
h_fc2 = intermediate_fc2

## softmax logic layer ##
prediction = tf.nn.softmax(h_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=h_fc2, labels=ys))
tf.add_to_collection('losses', cross_entropy)

avg_loss = tf.add_n(tf.get_collection('losses'))

train_step1 = tf.train.AdamOptimizer(lr1).minimize(avg_loss)
train_step2 = tf.train.AdamOptimizer(lr2).minimize(avg_loss)
train_step3 = tf.train.AdamOptimizer(lr3).minimize(avg_loss)
train_step4 = tf.train.AdamOptimizer(lr4).minimize(avg_loss)

##################################################################

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()

acc_mat = []
acc_iter = []
loss_mat = []
loss_iter = []

for i in range(iterration):
            
    batch_xs, batch_ys = cifar10.train.next_batch(batch_size, shuffle=True, flip=True, whiten=True, noise=False,crop=True,crop_test=False)
    
    #变化学习率
    if(0<= i <= 500):
        sess.run(train_step1, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    elif(500 < i <=3000):
        sess.run(train_step2, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    elif(3000 < i <=6000):
        sess.run(train_step3, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})        
    else:
        sess.run(train_step4, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    
    if i % 100 == 0:
        batch_xt, batch_yt = cifar10.test.next_batch(1280, shuffle=True, flip=False, whiten=True, noise=False,crop=False,crop_test=True)
        
        Test_Accuracy = compute_accuracy(batch_xt, batch_yt)
        acc_mat.append(Test_Accuracy)
        acc_iter.append(i)
        print("Test Accuracy:",Test_Accuracy)
        
        Training_Loss = compute_loss(batch_xs, batch_ys)
        loss_mat.append(Training_Loss)
        loss_iter.append(i)
        print("Training Loss:",Training_Loss)
        
        saver_path = saver.save(sess,"model/model.ckpt")
        print("model has been saved.")
        print("Time:",time.asctime(time.localtime(time.time())))
        
        #绘制loss曲线
        plt.title('Training Loss')
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.plot(loss_iter,loss_mat,'r',label='change of loss')
        plt.grid()
        plt.show()
        
        #绘制准确率曲线
        plt.title('Test Accuracy')
        plt.xlabel('iter')
        plt.ylabel('Accuracy')
        plt.plot(acc_iter,acc_mat,'b',label='change of Accuracy')
        plt.grid()
        plt.show()        
        
    print("Iteration:",i)

sess.close()