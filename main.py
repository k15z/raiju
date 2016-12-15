import numpy as np
import tensorflow as tf
from model import *
from tqdm import tqdm
from data import generator

def get_yc(s, a, r, ns, na, t, nt):
	yc = sess.run(y_out, feed_dict={x_in: s, t_in: t})
	yn = sess.run(y_out, feed_dict={x_in: ns, t_in: nt})
	for i in range(yc.shape[0]):
		yc[a[i]] = r[i] + 0.9 * yn[i,na[i]]
	return yc

writer = tf.summary.FileWriter('log', sess.graph)
saver = tf.train.Saver()
#saver.restore(sess, "./session.0.ckpt")

i = 0
for epoch in range(8):
	errs = []
	for s, a, r, ns, na, t, nt in tqdm(generator()):
		yc = get_yc(s, a, r, ns, na, t, nt)
		summary, err, _ = sess.run([stats, mse, step], feed_dict={x_in: s, y_exp: yc, t_in: t})
		writer.add_summary(summary, i)
		errs.append(err)
		if i % 1000 == 0:
			saver.save(sess, "./session." + str(epoch) + ".ckpt")
		i += 1
	print(epoch, sum(errs) / len(errs))
	saver.save(sess, "./session." + str(epoch) + ".ckpt")
