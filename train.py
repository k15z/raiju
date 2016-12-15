import numpy as np
import tensorflow as tf
from os import path
from tqdm import tqdm
from data import generator
from model import save, restore, sess, stats, train, x_in, y_out, y_exp


DATA_DIR = path.dirname(path.abspath(__file__))

def compute_targets(state, action, reward, nstate):
	batch_size = state.shape[0]
	y_target = sess.run(y_out, feed_dict={x_in: state})
	y_next = sess.run(y_out, feed_dict={x_in: nstate})
	for i in range(batch_size):
		y_target[action[i]] = reward[i] + 0.9 * np.max(y_next[i])
	return y_target

restore()
time = 0
writer = tf.summary.FileWriter(DATA_DIR + '/tensorboard', sess.graph)
for state, action, reward, nstate in tqdm(generator(batch_size=256)):
	y_target = compute_targets(state, action, reward, nstate)
	summary, _ = sess.run([stats, train], feed_dict={x_in: state, y_exp: y_target})
	writer.add_summary(summary, time)
	time += 1
	if time % 1000 == 999:
		save()
save()
