import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def csv_read(file_list):
    '''
    csv文件读取案例
    :param file_list:文件路径 + 文件名
    :return:读取的内容
    '''
    # 2 构造文件队列
    file_queue = tf.train.string_input_producer(file_list)

    # 3 文件阅读器
    reader = tf.TextLineReader()

    # 4 读取内容---构造csv阅读器读取队列数据（按一行）
    key, value = reader.read(file_queue)
    print(f'文件阅读器中key:{key},value:{value}')
    # 文件内容解码
    # record_defaults指定每一个样本每一列的类型, 指定默认值   [['None'],[1],[1.0]] ----> str, int, float类型
    record = [['None'],['None']]
    example, target = tf.decode_csv(value, record_defaults=record)

    # 5、批处理---读取多个数据进行批处理
    example_batch, target_batch = tf.train.batch([example, target], batch_size=9, num_threads=1, capacity=10)

    return example_batch, target_batch


if __name__ == '__main__':
    # cxv_read(file_list=123)

    # 读取文件名
    file_name = os.listdir('./csvdata')
    # 读取文件目录
    file_list = [os.path.join('./csvdata/', filename) for filename in file_name]

    print(file_list)
    # 1、找到文件，构造文件名传入列队中 文件路径 + 文件名
    example, target = csv_read(file_list)

    # 开启会话
    with tf.Session() as sess:

        # 开启线程协调器(回收)
        coord = tf.train.Coordinator()

        # 收集开启所有的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 输出在文件中所得到的内容
        print(f'{example},{target}')
        print(f'{example.eval()},{target.eval()}')
        print(sess.run([example, target]))

        # 回收子线程
        coord.request_stop()
        coord.join(threads)

