import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'



def picture_read(file_list):
    '''
    图片文件的读取案例
    :param file_list: 文件路径 + 文件名
    :return: 图片的内容
    '''
    # 1 构建图片列队
    file_queue = tf.train.string_input_producer(file_list)
    
    # 2 构建图片阅读器(默认一次读取一张图片)
    readers = tf.WholeFileReader()
    # 读取图片
    key, value = readers.read(file_queue)
    print(f'读取图片时的value:{value}')

    # 3 构建解码器
    image_encode = tf.image.decode_jpeg(value)
    print(image_encode)

    # 3.1 统一图片的大小       统一大小为[200, 200]
    # 注意：一定要把样本的形状固定 [200, 200, 3],在批处理的时候要求所有数据形状必须定义
    images_resize = tf.image.resize_images(image_encode,[200, 200])
    print(images_resize)

    images_resize.set_shape([200, 200, 3])          # 静态形状改变无返回值
    # images_reshape = tf.reshape(images_resize,[200, 200, 3])          # 采用动态形状改变形状
    print(f'统一图片大小之后的images_resize{images_resize}')

    # 4 批处理
    images_batch = tf.train.batch([images_resize], batch_size=10, num_threads=1, capacity=10)
    print(f'批处理之后的images_batch:{images_batch}')

    return images_batch

if __name__ == '__main__':

    # 获取图片的名字
    file_name = os.listdir('./picture')
    # print(file_name)

    # 获取图片路径 + 图片名
    file_list = [os.path.join('./picture',filename) for filename in file_name]

    images = picture_read(file_list)
    print(images)
    # 开启会话
    with tf.Session() as sess:
        # pass
        # 线程协调器
        coord = tf.train.Coordinator()

        # 获取所有子线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        print(sess.run(images))

        # 回收线程
        coord.request_stop()
        coord.join(threads)


