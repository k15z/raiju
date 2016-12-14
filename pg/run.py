import numpy as np
import tensorflow as tf

x_in = tf.placeholder(tf.float32, shape=(None, 30, 30, 3), name='x_in')
y_exp = tf.placeholder(tf.float32, shape=(None, 5), name='y_exp')
y_weight = tf.placeholder(tf.float32, shape=(None, 1), name='y_weight')

with tf.name_scope('conv1'):
    W = tf.Variable(tf.random_normal(
        [4, 4, 3, 16], 
        stddev=1.0 / tf.sqrt(float(4*4*3))),
    	name='weights'
    )
    h = tf.nn.conv2d(x_in, W, strides=[1, 1, 1, 1], padding='SAME')
    h = tf.nn.relu(h)

with tf.name_scope('conv2'):
    W = tf.Variable(tf.random_normal(
        [8, 8, 16, 32], 
        stddev=1.0 / tf.sqrt(float(8*8*16))),
    	name='weights'
    )
    h = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding='SAME')
    h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h = tf.nn.relu(tf.reshape(h, [-1, 15*15*32]))

W = tf.Variable(tf.random_normal(
    [15*15*32, 5], 
    stddev=1.0 / tf.sqrt(float(15*15*32)), name='weights'
))
b = tf.Variable(tf.zeros([5]), name='biases')
y_out = tf.matmul(h, W) + b

loss = tf.nn.softmax_cross_entropy_with_logits(y_out, y_exp)
train = tf.train.RMSPropOptimizer(0.01).minimize(tf.multiply(loss, y_weight))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, "./session.ckpt")

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_actions(states):
    acts = []
    dist = sess.run(y_out, feed_dict={x_in: states})
    for i in range(dist.shape[0]):
        print(dist[i])
        acts.append(np.random.choice(5, size=1, p=softmax(dist[i])))
    return acts

import data
S, A, R = next(data.generator())
for a, b, c in zip(get_actions(S), A, R):
	print(a, b, c)
