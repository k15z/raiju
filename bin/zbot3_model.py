import numpy as np
import tensorflow as tf

x_in = tf.placeholder(tf.float32, shape=(None, 30, 30, 3), name='x_in')
y_exp = tf.placeholder(tf.float32, shape=(None, 5), name='y_exp')

with tf.name_scope('conv1'):
    W = tf.Variable(tf.random_normal(
        [4, 4, 3, 8], 
        stddev=1.0 / tf.sqrt(float(4*4*3))),
        name='weights'
    )
    h = tf.nn.conv2d(x_in, W, strides=[1, 1, 1, 1], padding='SAME')
    h = tf.nn.elu(h)

with tf.name_scope('conv2'):
    W = tf.Variable(tf.random_normal(
        [6, 6, 8, 16], 
        stddev=1.0 / tf.sqrt(float(6*6*8))),
        name='weights'
    )
    h = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding='SAME')
    h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h = tf.nn.elu(tf.reshape(h, [-1, 15*15*16]))

with tf.name_scope('value'):
    W = tf.Variable(tf.random_normal(
        [15*15*16, 1], 
        stddev=1.0 / tf.sqrt(float(15*15*16)), name='weights'
    ))
    b = tf.Variable(tf.zeros([1]), name='biases')
    v_out = tf.matmul(h, W) + b

with tf.name_scope('action'):
    W = tf.Variable(tf.random_normal(
        [15*15*16, 5], 
        stddev=1.0 / tf.sqrt(float(15*15*16)), name='weights'
    ))
    b = tf.Variable(tf.zeros([5]), name='biases')
    a_out = tf.matmul(h, W) + b

y_out = a_out + v_out

mse = tf.reduce_mean(tf.square(y_out - y_exp))
step = tf.train.GradientDescentOptimizer(0.001).minimize(mse)

tf.summary.scalar('mse', mse)
stats = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, "./session.ckpt")

def get_actions(states):
    y_result = sess.run(y_out, feed_dict={x_in: states})
    return np.argmax(y_result, axis=1)
