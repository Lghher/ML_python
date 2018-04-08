#-*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np

## Save to file
# W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weight')
# b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')
#
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess,"C:/Users/hanghang/Desktop/1/save_net.ckpt")
#     print("Save to path: ",save_path)
#

# restore variables
W = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights')
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"C:/Users/hanghang/Desktop/1/save_net")
    print("biases:",sess.run(b))
    print("weights:", sess.run(W))