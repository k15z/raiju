import numpy as np
import tensorflow as tf
from model import *
from tqdm import tqdm
from data import generator

sess, y_out, x_in, mse, step, stats, y_exp = make_model()
writer = tf.summary.FileWriter('log', sess.graph)

saver = tf.train.Saver()
saver.restore(sess, "./session.ckpt")

def get_yc(s, a, r, ns):
	yc = sess.run(y_out, feed_dict={x_in: s})
	yn = sess.run(y_out, feed_dict={x_in: ns})
	for i in range(yc.shape[0]):
		yc[a[i]] = r[i] + 0.5 * np.max(yn[i])
	return yc

i = 0
for epoch in range(16):
	errs = []
	for s, a, r, ns in tqdm(generator(batch_size=128)):
		yc = get_yc(s, a, r, ns)
		summary, err, _ = sess.run([stats, mse, step], feed_dict={x_in: s, y_exp: yc})
		writer.add_summary(summary, i)
		errs.append(err)
		i += 1
	print(epoch, sum(errs) / len(errs))
	saver.save(sess, "./session.ckpt")
