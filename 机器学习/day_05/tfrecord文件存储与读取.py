import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# https://www.cs.toronto.edu/~kriz/cifar.html
'''
CIFAR-10数据集由10个类中的60000个32x32彩色图像组成，每个类有6000个图像。有50000个训练图像和10000个测试图像。 

二进制版
二进制版本包含文件data_batch_1.bin，data_batch_2.bin，...，data_batch_5.bin以及test_batch.bin。
每个文件的格式如下：
<1 x label> <3072 x pixel>
...
<1 x label> <3072 x pixel>
换句话说，第一个字节是第一个图像的标签，它是0-9范围内的数字。接下来的3072个字节是图像像素的值。
前1024个字节是红色通道值，下一个1024是绿色，最后1024个是蓝色。值以行主顺序存储，因此前32个字节是图像第一行的红色通道值。 

每个文件包含10000个这样的3073字节“行”图像，尽管没有划分行的任何内容。因此，每个文件应该是30730000字节长。 

还有另一个名为batches.meta.txt的文件。这是一个ASCII文件，它将0-9范围内的数字标签映射到有意义的类名。
它只是10个类名的列表，每行一个。第i行上的类名对应于数字标签i。
'''


# 定义cifar的数据等命令行参数
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("cifar_dir", "./cifar-10-binary/cifar-10-batches-bin/", "文件的目录")
tf.app.flags.DEFINE_string("cifar_tfrecords", "./tfrecord_test/cifar.tfrecords", "存进tfrecords的文件")
tf.app.flags.DEFINE_string("cifar_read_tfrecords", "./tfrecord_test/", "tfrecords的文件目录")



class binary_file_read():

    def __init__(self,file_list):
        self.file_list = file_list
        self.length = 32
        self.width = 32
        self.channel = 3
        self.label_byte = 1
        self.image_byte = self.length * self.width * self.channel
        self.one_sample_bytes = self.label_byte + self.image_byte
        pass



    def read_tfrecord(self):
        '''
        读取tfrecord文件
        :return:None
        '''
        # 1.构建文件列队
        file_queue = tf.train.string_input_producer(self.file_list)

        # 2.构建tfrecord文件阅读器
        # 构造文件阅读器，读取内容example,value=一个样本的序列化example
        reader = tf.TFRecordReader()
        key, value = reader.read(file_queue)
        print(f'文件阅读其中的key:{key},value:{value}')

        # (较之于其他类型文件的读取多出来的一个步骤)
        # 3.解析example协议块
        features = tf.parse_single_example(value, features={
            'label':tf.FixedLenFeature([],tf.int64),
            'image':tf.FixedLenFeature([],tf.string)
        })

        # 4.构建tfrecord文件解码器
        # 解码内容, 如果读取的内容格式是string需要解码， 如果是int64, float32不需要解码
        # 其中image为string格式，label为int64格式
        image = tf.decode_raw([features['image']], tf.uint8)
        print(f'解析解码之后值label:{features["label"]},image:{image}')

        # 4.1形状的固定
        image_reshape = tf.reshape(image, [self.length, self.width, self.channel])

        # 5.批处理
        label_batch, image_batch = tf.train.batch([features["label"],image_reshape], batch_size=10, num_threads=1, capacity=10)
        print(f'批处理之后的label_bach:{label_batch},image_batch:{image_batch}')

        return label_batch, image_batch



if __name__ == '__main__':

    # 读取文件
    file_name = os.listdir(FLAGS.cifar_read_tfrecords)
    file_list = [os.path.join(FLAGS.cifar_read_tfrecords,file) for file in file_name]

    read_bin_file = binary_file_read(file_list)
    label_batch, image_batch = read_bin_file.read_tfrecord()
    with tf.Session() as sess:

        # 线程协调器
        coord = tf.train.Coordinator()

        # 开启线程操作
        threads = tf.train.start_queue_runners(sess,coord=coord)

        # 运行操作--训练等
        print(f'输出label_batch:{sess.run(label_batch)}')
        print(f'输出image_batch:{sess.run(image_batch)}')

        # 回收子线程
        coord.request_stop()
        coord.join(threads)


