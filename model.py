"""
This uses a convolutional neural network to implement offline Q-learning.
===============================================================================
Copyright 2016, Kevin Zhang <kevz@mit.edu>
"""
import tensorflow as tf

x_in = tf.placeholder(tf.float32, shape=(None, 16, 16, 3), name='x_in')
y_exp = tf.placeholder(tf.float32, shape=(None, 5), name='y_exp')

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
        [4, 4, 16, 32], 
        stddev=1.0 / tf.sqrt(float(4*4*16))),
    	name='weights'
    )
    h = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding='SAME')
    h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h = tf.nn.relu(tf.reshape(h, [-1, 8*8*32]))

with tf.name_scope('output'):
    W = tf.Variable(tf.random_normal(
        [8*8*32, 5], 
        stddev=1.0 / tf.sqrt(float(8*8*32)), name='weights'
    ))
    b = tf.Variable(tf.zeros([5]), name='biases')
    y_out = tf.matmul(h, W) + b

mse = tf.reduce_mean(tf.square(y_out - y_exp))
step = tf.train.RMSPropOptimizer(0.001).minimize(mse)

tf.summary.scalar('mse', mse)
stats = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('log', sess.graph)

if __name__ == "__main__":
	import numpy as np
	from tqdm import tqdm
	from data import generator

	def get_yc(s, a, r, ns):
		yc = sess.run(y_out, feed_dict={x_in: s})
		yn = sess.run(y_out, feed_dict={x_in: ns})
		for i in range(yc.shape[0]):
			yc[a[i]] = r[i] + 0.5 * np.max(yn[i])
		return yc

	i = 0
	for epoch in range(32):
		errs = []
		for s, a, r, ns in tqdm(generator(batch_size=128)):
			yc = get_yc(s, a, r, ns)
			summary, err, _ = sess.run([stats, mse, step], feed_dict={x_in: s, y_exp: yc})
			writer.add_summary(summary, i)
			errs.append(err)
			i += 1
		print(epoch, sum(errs) / len(errs))
