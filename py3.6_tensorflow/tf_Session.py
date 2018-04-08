#-*- coding=utf-8 -*-
import tensorflow as tf

matirx1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

product = tf.matmul(matirx1,matrix2)

# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
