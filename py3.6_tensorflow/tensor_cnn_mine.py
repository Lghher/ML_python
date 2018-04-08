#-*- coding=utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
# 读取MNIST数据
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

### define function ###

# product weight tensor
def weight_variable(shape):
    # truncated_normal 该函数随机产生的正态分布，每个值与平均值的差不会超过两个标准差
    initial = tf.truncated_normal(shape,mean=0,stddev=0.1)
    return tf.Variable(initial)

# product bias tensor
def bias_variable(shape):
    # 创建一个shape为输入shape的矩阵，所有值都为0.1
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    # strides为[1,x_stride,y_stride,1]第一个和第四个必须为1，padding='SAME'为filter不到边缘的时候边缘补零
    # W为filter参数
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    # strides为[1,x_stride,y_stride,1]第一个和第四个必须为1，padding='SAME'为filter不到边缘的时候边缘补零
    # ksize为池化层的kernel的大小,第一个和第四个必须为1
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


# 定义输入输出结构
# 输入
# None表示输入图片的数量不定
xs = tf.placeholder(tf.float32,[None,28*28])
# 类别数0-9  10个
ys = tf.placeholder(tf.float32,[None,10])
# keep_prob是dropout的参数，为不删除掉节点的比例
keep_prob = tf.placeholder(tf.float32)
# x_# x_image又把xs reshape成了28*28*1的形状，因为是灰色图片，所以通道是1.
# 作为训练时的input，-1代表图片数量不定
x_image = tf.reshape(xs,[-1,28,28,1])


# build nerual network , 也就是前向的计算公式

# 第一层卷积
# 第一二参数值是卷积核尺寸大小，即patch，
# 第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征图像，代表卷积后的图像通道数
W_conv1 = weight_variable([5,5,1,32])
# 每个卷积核都对应一个偏置
b_conv1 = bias_variable([32])
# 图片乘以卷积核，并加上偏执量，卷积结果28x28x32
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
# 池化结果14x14x32 卷积结果乘以池化卷积核
h_pool1 = max_pool_2x2(h_conv1)
# 第二层卷积
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
# 注意h_pool1是第一层的池化结果，第二层卷积结果为14x14x64
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
# 池化结果为7x7x64
h_pool2 = max_pool_2x2(h_conv2)

# 第三层为全连接层
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
# 将第二层卷积结果h_pool2 reshape为一行7*7*64的数据[n_samples,7,7,64] > [n_samples,7*7*64]
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

# 在全连接层进行dropout，减少过拟合
h_fc1_dropout = tf.nn.dropout(h_fc1,keep_prob)

# 第四层是输出层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
# 最后的分类结果为n_samples*10  softmax是多分类
y_conv = tf.nn.softmax(tf.matmul(h_fc1_dropout,W_fc2)+b_fc2)
# loss function 是 交叉熵
cross_entropy = -tf.reduce_sum(ys * tf.log(y_conv))
# 学习率个人感觉1e-3还是比较好的，高一点的效果都不好，用的是cpu版本的tf，跑的死慢，
# 迭代100步要个15s，找机会要启动我6g的1060了，不过最后效果还是可以跑到0.98的准确率
train = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)

#####开始训练

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(20000):
        batch = mnist.train.next_batch(50)
        if step % 100 == 0:
            print(sess.run(accuracy,feed_dict={xs:batch[0],ys:batch[1],keep_prob:0.5}))
        sess.run(train,feed_dict={xs:batch[0],ys:batch[1],keep_prob:0.5})
    print(sess.run(accuracy,feed_dict={xs:mnist.test.images,ys:mnist.test.labels,keep_prob:0.5}))
