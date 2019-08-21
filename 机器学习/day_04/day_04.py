import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def draw_graph():
    '''
    图的构建,张量(tensor)与运算操作(Operation)的说明

    :return:
    '''
    # 创建一张图包含了一组op和tensor,上下文环境
    # op:只要使用TensorFlow的API定义的函数都是OP, 即constant也是一个OP
    # tensor:就指代的是数据, constant中存在的数据就称之为tensor(张量)

    g = tf.Graph()

    print(g)
    with g.as_default():
        c = tf.constant(11.0)
        print(c.graph)


    # 实现一个加法运算
    # 这里定义的是一个算法的一个大框架,即在一张图之中
    a = tf.constant(5.0)
    b = tf.constant(6.0)
    # print(a.eval())
    sum1 = tf.add(a,b)

    # 不是op 不能运行
    # var1 = 1
    # var2 = 2
    # sum2 = var1 + var2

    # 有重载机制,将运算符转化为op类型
    var3 = 1.0
    sum3 = a + var3


    print(sum1)

    graph = tf.get_default_graph()

    print(graph)
    # 算法的基本框架,在一张图之中

    # 训练模型
    # 实时的提供数据去进行训练(实时训练是用的到)
    plh = tf.placeholder(tf.float32,[None,3])       # 可以接受 任意行,但是特征为3个
    print(plh)

    # 只能运行一个图结构,可以在会话中指定图
    # with tf.Session(graph=g) as sess
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess 多一个图中的张量查看
    with tf.Session() as sess:      # 绘画
        print(sess.run(sum1))
        print(sum1.eval())      # eval() == sess.run
        print(sum3.eval())
        # 实时训练时,实时提供数据
        print(sess.run(plh,feed_dict={plh:[[1,2,3],[2,3,4],[3,4,5]]}))
        print(a.graph)
        print(sum1.graph)
        print(sess.graph)
    # 报错经常出现于会话中
    '''
    返回值异常
	RuntimeError：如果它Session处于无效状态（例如已关闭）。
	TypeError：如果fetches或feed_dict键是不合适的类型。
	ValueError：如果fetches或feed_dict键无效或引用 Tensor不存在。

    '''

    # TensorFlow:打印出来的形状表示
    # 0维:()     1维:(5)      2维:(2,3)       3维:(2,3,4)--2张(3,4)的表


    return None

def change_shape():
    '''
    形状的概念
    静态形状和动态形状
    1、对于静态形状,一旦张量形状固定了,不能再次设置静态形状       1D->1D   2D->2D
    2、动态形状可以去创建一个新的张量,改变时候一定要注意元素数量要匹配  1D->2D
    3、静态形状不能跨纬度修改
    :return: None
    '''
    plh = tf.placeholder(tf.float32,[None,2])
    print(plh)

    plh.set_shape([3,2])
    print(plh)

    # 动态变量
    plh_reshape = tf.reshape(plh,[2,3])
    print(plh_reshape)      # 类型发生改变"reshape"

def tensor_product():
    '''
    张量API,生成张量
    https://www.tensorflow.org/versions/r1.0/api_guides/python/math_ops
    算术运算符
    基本数学函数
    矩阵运算
    减少维度的运算(求均值)
    序列运算
    :return:
    '''
    zeros = tf.zeros([3,4],tf.float32)
    ones = tf.ones([3,4],tf.float32)
    random_tensor = tf.random_normal([2,3], mean=0, stddev=1.0, dtype=tf.float32)

    # 类型的转化
    # [1,2,3]为int类型将其转化为float类型
    trans_a = tf.cast([1,2,3],tf.float32)

    # 张量合并
    combine_a = [[1,2,3],[2,3,4]]
    combine_b = [[4,5,6],[5,6,7]]
    combine_num = tf.concat([combine_a,combine_b],axis=1)


    with tf.Session() as sess:
        print(zeros.eval())
        print(ones.eval())
        print(sess.run(random_tensor))
        print(sess.run(trans_a))
        print(sess.run(combine_num))

def op():
    '''
    变量
    1、变量op能够持久化保存,普通张量op是不行的
    2、当定义一个变量op时,需要先一步做显示初始化操作
    3、name参数:在tensorboard使用的时候显示名字,可以让相同op名字的进行区分
    terminal 中开启
    tensorboard    --logdir=/tmp/tensorflow/summary/test/

    :return: None
    '''
    # 常数张量
    a = tf.constant([1,2,3,4,5], name = 'a')
    var = tf.Variable(tf.random_normal([2,3], mean=0, stddev=1.0), name='variable')
    print(a,'\n',var)

    cons_1 = tf.constant(1.0, name='cons_1')
    cons_2 = tf.constant(2.0, name='cons_2')

    sum = tf.add(cons_1, cons_2, name='add')

    # 变量必须进一步做显示的初始化
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        # 把程序的图结构入时间文件 tensorboard可视化
        # 此处保存的tensorboard可视化文件时, 保存的位置地址写为"绝对地址",相对地址容易出错
        filewriter = tf.summary.FileWriter('D:\pycharm-xx\机器学习\day_04\summary',graph=sess.graph)

        print(sess.run(sum))


    return None

def linerRegression_tf():
    '''
    TensorFlow实现线性回归预测
    # 1、训练参数问题:trainable
    # 学习率和步数的设置：

    # 2、添加权重参数，损失值等在tensorboard观察的情况 1、收集变量2、合并变量写入事件文件
    :return:
    '''
    # # 第一个参数：名字，默认值，说明
    # tf.app.flags.DEFINE_integer("max_step", 100, "模型训练的步数")
    # tf.app.flags.DEFINE_string("model_dir", " ", "模型文件的加载的路径")
    #
    # # 定义获取命令行参数名字
    # FLAGS = tf.app.flags.FLAGS

    with tf.variable_scope('data'):
        # 1 准备数据, x 特征值 [100,1]   y 目标值[100]
        x_data = tf.random_normal([100,1], mean=1.75, stddev=0.5, name='x_data',dtype=tf.float32)       # 能书写数据类型的都写上, 不然会有意想不到的错误

        # 矩阵相乘必须是二维的
        y_ture = tf.matmul(x_data,[[2.0]],name='y_ture') + 0.8        # 注意这里的的数据为浮点数float32 2.0

    with tf.variable_scope('model'):
        # 2 建立线性回归模型 1个特征 1个偏置 y = wx + b
        # 随机设置权重w, 与偏置b, 随后进行优化更新
        # 变量中trainable设置变量是否变化更新  训练参数
        weight = tf.Variable(tf.random_normal([1,1], mean=0.0, stddev=0.5, name='weight',dtype=tf.float32), trainable=True)
        bias = tf.Variable(1,name='bias',dtype=tf.float32)

    with tf.variable_scope('loss'):

        # 3 建立均方误差值loss
        y_predict = tf.matmul(x_data,weight) + bias

        loss = tf.reduce_mean(tf.square(y_ture - y_predict))

    with tf.variable_scope('optimizer'):

        # 梯度下降优化损失函数  学习率取值 0~1 , 2, 3, 4, 5, 7...
        # 学习率过大会使随时函数发生震荡, 最终无法得到最优解
        opt_loss = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


    '''
    op初始化
    '''
    # 初始化变量op
    init_var = tf.global_variables_initializer()

    # 收集变量op (tensorboard中能更好的看出数据的变化过程)
    with tf.variable_scope('collect_op'):
        tf.summary.scalar('loss',loss)
        tf.summary.histogram('weight',weight)
        merge_op = tf.summary.merge_all()

    save_model = tf.train.Saver()

    # 开启会话
    with tf.Session() as sess:
        sess.run(init_var)
        # print(sess.run([x_data,y_ture]))

        print(f'初始随机开始的权重系数w:{weight.eval()},b:{bias.eval()}')

        # 建立事件文件
        filwriter = tf.summary.FileWriter('D:\pycharm-xx\机器学习\day_04\summary',graph=sess.graph)

        # 加载文件,覆盖模型中的随机定义的参数,从上次接受的参数开始训练
        if os.path.exists('D:\pycharm-xx\机器学习\day_04\ckpt\checkpoint'):
            save_model.restore(sess,'D:\pycharm-xx\机器学习\day_04\ckpt\model')


        # 进行梯度下降
        for i in range(200):
            # 运行优化
            sess.run(opt_loss)

            # 收集变量之后--> 合并变量,写入时间
            # 运行合并
            summary_write = sess.run(merge_op)
            filwriter.add_summary(summary_write,i)


            print(f'第{i}次优化之后的权重值w:{weight.eval()},b:{bias.eval()}')
        save_model.save(sess,'D:\pycharm-xx\机器学习\day_04\ckpt\model')


if __name__ == '__main__':
    # draw_graph()
    # change_shape()      # 形状的改变,经常使用需记忆
    # tensor_product()        # 张量生成的API
    # op()
    linerRegression_tf()












