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

    def binary_read(self):

        # 1.构造文件列队
        file_queue = tf.train.string_input_producer(self.file_list)

        # 2.构造二进制文件阅读器
        reader = tf.FixedLengthRecordReader(self.one_sample_bytes)
        key, value = reader.read(file_queue)
        print(key)
        print(value)
        # 2.1 犹豫二进制文件中特征与标签值是混在一起 ,故需要将其分开


        # 3.构造二进制文件解码器
        decode_value = tf.decode_raw(value, tf.uint8)
        print(f'文件内容解码时value:{decode_value}')

        ''' 尝试性行为
        # 3.1制定decode_value的形状，仅固定了形状大小
        reshape_value = tf.reshape(decode_value,[32, 32, 3])
        print(f'动态形状改变之后的value:{reshape_value}')
        '''
        # 3.2 划分特征值和目标值 ----分割出图片和标签数据，切除特征值和目标值
        label_value = tf.cast(tf.slice(decode_value, [0], [self.label_byte]), tf.int32)
        image_value = tf.slice(decode_value, [self.label_byte], [self.image_byte])
        print(f'划分target:{label_value},image_balue:{image_value}')

        # 3.2.1将image_value 变化形状(动态变化)  可以对图片的特征数据进行形状的改变 [3072] --> [32, 32, 3]
        image_value = tf.reshape(image_value, [32, 32, 3])
        print(f'变化形状之后的image_value:{image_value}')

        # 4.进行批处理
        label_value_batch, image_vale_batch = tf.train.batch([label_value, image_value], batch_size=10, num_threads=1, capacity=10)
        print(f'进行批处理之后的label_value_batch, image_vale_batch:{label_value_batch, image_vale_batch}')


        return label_value_batch, image_vale_batch

    def tfrecord_write(self,label_value_batch,image_vale_batch):
        '''
        将二进制文件写入cifar_tfrecords中
        :return: None
        '''
        # 将二进制文件写入tfrecord文件中
        # 1.建立TFrecord存储器
        writer = tf.python_io.TFRecordWriter(FLAGS.cifar_tfrecords)

        # 2.构造每个样本的example协议块
        # 每批次读取10个
        for i in range(10):
            # 读取样本中的第i个图片
            '''
            需要序列化iamge转化为字符串
            '''
            label = int(label_value_batch[0].eval()[0])
            image = image_vale_batch[0].eval().tostring()
            # print('writer',label,image)

            # 构造一个样本的example
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

            }))

            writer.write(example.SerializeToString())
        writer.close()
        return None

if __name__ == '__main__':

    # 读取文件
    file_name = os.listdir(FLAGS.cifar_dir)
    file_list = [os.path.join(FLAGS.cifar_dir,file) for file in file_name if file[-3:] == 'bin']

    read_bin_file = binary_file_read(file_list)
    label_value_batch, image_vale_batch = read_bin_file.binary_read()

    with tf.Session() as sess:

        # 线程协调器
        coord = tf.train.Coordinator()

        # 开启线程操作
        threads = tf.train.start_queue_runners(sess,coord=coord)

        # 运行操作--训练等
        print(f'输出的label_value_batch:{label_value_batch.eval()}')
        print(f'输出的image_vale_batch:{image_vale_batch.eval()}')


        # 将二进制文件写入tfrecord文件中
        print('开始写入tfrecord中')
        read_bin_file.tfrecord_write(label_value_batch, image_vale_batch)
        print('完成写入')


        # 回收子线程
        coord.request_stop()
        coord.join(threads)


