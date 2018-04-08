#-*- coding=utf-8 -*-
import tensorflow as tf

##########   操作间所传递的数据都可以看做是Tensor ##########
m1 = tf.Variable([3,5],dtype=tf.float32)
m2 = tf.Variable([2,4],dtype=tf.float32)
print(m1)
result = tf.add(m1,m2)
print(result)
sess = tf.Session()

# 有Variable定义的时候必须先run这一步
sess.run(tf.global_variables_initializer())
print(sess.run(result))
sess.close()
with tf.Session() as sess:
    m1 = tf.constant([1,2],dtype=tf.float32)
    m2 = tf.constant([1,1],dtype=tf.float32)
    result = tf.add(m1,m2)
    print(sess.run(result))

sess = tf.Session()
#python只有函数 类中才有新作用域，所以下面使用的result是with里面的
# print(sess.run(result))
a = tf.constant([[2.,3.]],name="a")
b = tf.constant([[1.],[4.]],name="b")
result = tf.matmul(a,b,name="mul")
print(result)
print(a)
print(sess.run(result))

a = tf.zeros([2,2],dtype=tf.float32)
b= tf.zeros_like(a,optimize=True)
print(sess.run(a))
print(sess.run(b))

random_num = tf.random_normal([2,3],mean=-1,stddev=4,
                              dtype=tf.float32,seed=None,name='rnum')
print(sess.run(random_num))

a = tf.constant(5)
b = tf.constant(6)
c = tf.constant(4)
add =  tf.add(b,c)
mul = tf.multiply(a,add)
print(sess.run([mul,add]))

