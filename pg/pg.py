import data
import tqdm
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

for S, A, R in tqdm.tqdm(data.generator(batch_size=128)):
    batch_size = len(S)
    y_target = np.zeros((batch_size, 5))
    y_reward = np.zeros((batch_size, 1))
    for i in range(batch_size):
        y_target[i, A[i]] = 1.0
        y_reward[i, 0] = 0.01*R[i]
    sess.run(train, feed_dict={x_in: S, y_exp: y_target, y_weight: y_reward})

saver = tf.train.Saver()
saver.save(sess, "./session.ckpt")
