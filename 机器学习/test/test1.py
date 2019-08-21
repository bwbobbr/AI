import tensorflow as tf

var1 = tf.Variable(tf.truncated_normal([2,3], mean=0.0, stddev=1.0, dtype=tf.float32))


init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(var1.eval())