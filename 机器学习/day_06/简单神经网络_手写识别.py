'''
Mnist书写数字识别
28 * 28 = 784 像素(特征值)
mnist.train 与 mnist.test
mnist.train.images : 训练数据集
mnist.train.labels : 训练数据集的标签
数据集下载地址:http://yann.lecun.com.exdb/mnist/
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# mnist = input_data.read_data_sets('./mnist/input_data', one_hot=True)
# # print(mnist)

# 定义命令行参数
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('is_train',1,'是否进行训练')

def fully_connected():
    '''
    神经网络的简单考虑,仅仅考虑一个全连接层
    :return: None
    '''
    mnist = input_data.read_data_sets('./mnist/input_data', one_hot=True)

    # 1.建立数据的占位符 x [None, 784] 7_ture [None, 10]
    with tf.variable_scope('data'):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.float32, [None, 10])

    # 2.初始化权重和偏执参数 w 与 bias
    with tf.variable_scope('init_parameter'):
        weight = tf.Variable(tf.random_normal([784, 10], mean=0, stddev=1.0, name='weight'))
        bias = tf.Variable(tf.constant(0.0, tf.float32, shape=[10]))

    # 3.搭建模型(神经网络--单全连接层)
    with tf.variable_scope('fully_connected_model'):
        y_predict = tf.matmul(x, weight) + bias
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))
        # 注意此处的loss计算出来的是平均误差

    # 4.进行反向传播优化算法(本质上依旧是梯度下降优化算法)
    with tf.variable_scope('gradient_opt'):
        grad_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 5.准确率的计算
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))


    # 增加变量显示
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', accuracy)
    tf.summary.histogram('weight', weight)
    tf.summary.histogram('bias', bias)

    # 进行变量初始化与会话前的操作
    init_op = tf.global_variables_initializer()

    # 定义一个合并变量的 op
    merge_summary = tf.summary.merge_all()

    # 创建一个saverexecfile
    saver = tf.train.Saver()


    # 开启会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 建立事件文件
        filewriter = tf.summary.FileWriter('./summary/', graph=sess.graph)

        # 判断程序是进行训练还是进行预测结果
        if FLAGS.is_train == 1:
            for i in range(2000):

                # 读取真实存在的特征值和目标值
                mnist_x, mnist_y = mnist.train.next_batch(50)
                sess.run(grad_op, feed_dict={x: mnist_x, y_true: mnist_y})

                summary = sess.run(merge_summary, feed_dict={x: mnist_x, y_true: mnist_y})
                filewriter.add_summary(summary,i)

                print(f'训练第{i}次,准确率为:{sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})}')

            # 保存该训练的模型
            saver.save(sess, './ckpt/mnist_model')
        else:
            saver.restore(sess, './ckpt/mnist_model')

            for i in range(100):
                x_test, y_test = mnist.test.next_batch(1)
                y_test_true = tf.argmax(y_test,1).eval()
                y_test_predict = tf.argmax(sess.run(y_predict, feed_dict={x: x_test, y_true: y_test}),1).eval()

                print(f'预测第{i}张图片,真实数值为{y_test_true}, 预测数值为{y_test_predict}')

    return None



if __name__ == '__main__':
    fully_connected()