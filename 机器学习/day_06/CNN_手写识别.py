from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


'''
Mnist书写数字识别
28 * 28 = 784 像素(特征值)
mnist.train 与 mnist.test
mnist.train.images : 训练数据集
mnist.train.labels : 训练数据集的标签
数据集下载地址:http://yann.lecun.com.exdb/mnist/
'''

def init_weight(shape):
    init_weight = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return init_weight

def init_bias(shape):
    init_bias = tf.Variable(tf.constant(0.0,shape=shape))
    return init_bias

def model():
    '''

    :return:
    '''
    # 准备数据的占位符 x [None, 784]  y_true [None, 10]
    with tf.variable_scope("data_holder"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.int32, [None, 10])

    # 第一层卷积层
    with tf.variable_scope("cov1"):
        '''
        conv: filter32, 5*5*1, strides=1, [None, 28, 28, 1]---->[None, 28, 28, 32]
        relu: tf.nn.relu, [None, 28, 28, 32]
        pooling: 2*2, strides=2, [None, 28, 28, 32]---->[None, 14, 14, 32]

        '''
        # 对x进行形状的改变[None, 784]  [None, 28, 28, 1]
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])

        # 初始化cov权重
        conv_filter1 = init_weight([5, 5, 1, 32])
        bias1 = init_bias([32])

        # conv层
        conv1 = tf.nn.conv2d(x_reshape, conv_filter1, strides=[1, 1, 1, 1], padding='SAME') + bias1

        # relu层
        relue1 = tf.nn.relu(conv1)

        # pooling层
        # pooling_filter1 = init_weight([2, 2, 1, 32])
        pooling1 = tf.nn.max_pool(relue1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    # 第二层卷积层
    with tf.variable_scope("cov1"):
        '''
        conv: filter64, 5*5*32, strides=1, [None, 14, 14, 32]---->[None, 14, 14, 64]
        relu: tf.nn.relu, [None, 14, 14, 64]
        pooling: 2*2, strides=2, [None, 14, 14, 64]---->[None, 7, 7, 64]

        '''
        conv_filter2 = init_weight([5, 5, 32, 64])
        bias2 = init_bias([64])
        # conv层
        conv2 = tf.nn.conv2d(pooling1, conv_filter2, strides=[1, 1, 1, 1], padding='SAME') + bias2

        # relu层
        relue2 = tf.nn.relu(conv2)

        # pooling层
        # pooling_filter1 = init_weight([2, 2, 1, 32])
        pooling2 = tf.nn.max_pool(relue2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 全连接层
    with tf.variable_scope("fc"):

        # 初始化权重
        fc_weight = init_weight([7*7*64, 10])
        fc_bias = init_bias([10])

        # 建立模型, 搭建模型(神经网络--单全连接层)
        with tf.variable_scope('fully_connected_model'):

            x_fc_reshape = tf.reshape(pooling2, [-1, 7*7*64])
            y_predict = tf.matmul(x_fc_reshape, fc_weight) + fc_bias
        #     # 计算损失
        #     softmax_cross = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict)
        #     loss = tf.reduce_mean(softmax_cross)
        #
        # with tf.variable_scope('gradient_opt'):
        #     # 优化, 进行反向传播优化算法(本质上依旧是梯度下降优化算法)
        #     opt = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
        #
        # # 准确率的计算
        # with tf.variable_scope("acc"):
        #     equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        #     accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 此处返回的占位符中的'x'与'y_true'值, 并非转化之后的x_fc_reshape值
    return x, y_true, y_predict


def conv_fc():
    '''
    CNN实现手写数字识别
    :return: None
    '''
    # 读取数据位置信息
    mnist = input_data.read_data_sets('./mnist/input_data', one_hot=True)

    x, y_true, y_predict = model()
    with tf.variable_scope("soft_cross"):
        # 计算损失
        softmax_cross = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict)
        loss = tf.reduce_mean(softmax_cross)

    with tf.variable_scope('gradient_opt'):
        # 优化, 进行反向传播优化算法(本质上依旧是梯度下降优化算法)
        opt = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

    # 准确率的计算
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))


    # 初始化变脸
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        for i in range(1000):
            mnist_x, mnist_y = mnist.train.next_batch(50)
            sess.run(opt, feed_dict={x: mnist_x, y_true: mnist_y})
            print(f'训练第{i}次,准确率为:{sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})}')

    return None



if __name__ == '__main__':
    conv_fc()