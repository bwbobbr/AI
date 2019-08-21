import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def imitate_synchronize():
    '''
    模拟一下同步先处理数据，然后才能取数据训练
    :return:
    '''
    # 1 创建一个队列
    queue = tf.FIFOQueue(1000, tf.float32)
    # 放入数据
    enq_m = queue.enqueue_many([[0.1, 0.3, 0.4],])      # 注意数据形式, [0.1, 0.3, 0.4]会被看做是一个张量, 而非多个数据

    # 2 进行读取数据, 处理数据的操作   读取数据之后--> +1 --> 放入数据
    enq_one = queue.dequeue()       # 读取数据
    enq_add = enq_one + 1           # 对数据进行操作
    deq_one = queue.enqueue(enq_add)      # 放回数据


    with tf.Session() as sess:
        # 初始化队列, 放入初始数据
        sess.run(enq_m)

        # 对数据进行操作
        for i in range(100):
            sess.run(deq_one)       # 连锁反应, 步步关联的计算, 只需要书写相关部分的最后一步

        for j in range(queue.size().eval()):        # 注意此处的queue.size() 为一个op 故, 需要eval() 将其变为数据
            print(sess.run(queue.dequeue()))

    return None


def imitate_asynchronize():
    '''
    模拟异步读取数据的过程
    模拟异步子线程 存入样本， 主线程 读取样本

    :return: None
    '''
    # 1 构建队列, 定义一个1000数据的队列
    queue = tf.FIFOQueue(1000, tf.float32)

    # 2 定义操作 +1 放入数据
    var = tf.Variable(0.0)
    data = tf.assign_add(var, tf.constant(1.0))
    deq_one = queue.enqueue(data)

    # 子线程, 队列管理器 ---定义队列管理器op, 指定多少个子线程，子线程该干什么事情
    qr = tf.train.QueueRunner(queue, enqueue_ops=[deq_one] * 2)

    # 初始化变量
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 初始化队列
        sess.run(init_op)

        # 开启线程协调器----用于回收线程
        coord = tf.train.Coordinator()

        # 定义子线程的操作, 开启子线程
        threads = qr.create_threads(sess,coord=coord, start=True)


        # 主线程读取数据, 训练
        for i in range(300):
            print(sess.run(queue.dequeue()))


        # 子线程的回收
        coord.request_stop()
        coord.join(threads)
        pass

    return None



if __name__ == '__main__':
    # imitate_synchronize()
    imitate_asynchronize()