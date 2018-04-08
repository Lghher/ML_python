#-*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def compute_accurary(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accurary = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accurary,feed_dict={xs:v_xs,ys:v_ys})
    return result

def weight_variable(shape):
    inital = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(inital)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,W):
    #stride[1,x_movement,y_movement,1]
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    # stride[1,x_movement,y_movement,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,28,28,1])

## conv1 layer ##
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

## conv2 layer ##
W_conv2 = weight_variable([5,5,32,64]) # patch 5x5,in size 32,out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

## func1 layer##
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

## func1 layer##

## func2 layer##
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

## func2 layer##




mnist = input_data.read_data_sets('MNIST_data',one_hot='True')
# def add_layer(inputs, in_size, out_size, activation_function=None):
#     with tf.name_scope('layer'):
#         with tf.name_scope('Weight'):
#             Weights = tf.Variable(tf.random_normal([in_size,out_size]))
#         with tf.name_scope('biases'):
#              biases = tf.Variable(tf.zeros([1,out_size])+ 0.1)
#         with tf.name_scope('Wx_plus_b'):
#             Wx_plus_b = tf.matmul(inputs,Weights) + biases
#         if activation_function is None:
#             outputs = Wx_plus_b
#         else:
#             outputs = activation_function(Wx_plus_b)
#         return  outputs
#
# prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)
#


cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                     reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i % 50 == 0:
        print(compute_accurary(mnist.test.images,mnist.test.labels))



