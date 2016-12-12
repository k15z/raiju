import json
import random
import numpy as np
from os import path
from glob import glob

DATA_DIR = path.dirname(path.abspath(__file__))

def get_reward(state):
	reward = 0.0
	for y in range(state.shape[0]):
		for x in range(state.shape[1]):
			reward += state[y,x,0] * state[y,x,1]
	return reward / (state.shape[0] * state.shape[1])

def get_ndarray(frame, production, num):
	width, height = len(frame[0]), len(frame)
	arr = np.zeros((height, width, 3))
	for y in range(height):
		for x in range(width):
			player, strength = frame[y][x]
			arr[y,x,0] = 1.0 if player == num else -1.0
			arr[y,x,1] = strength / 255.0
			arr[y,x,2] = production[y][x] / 255.0
	return arr

def parse_sars(frame, move, nframe, production, num):
	width, height = len(frame[0]), len(frame)
	arr = get_ndarray(frame, production, num)
	narr = get_ndarray(nframe, production, num)
	for y in range(height):
		for x in range(width):
			if frame[y][x][0] > 0.0:
				state = np.roll(np.roll(arr, -y+8, axis=0), -x+8, axis=1)[:16,:16,:]
				nstate = np.roll(np.roll(narr, -y+8, axis=0), -x+8, axis=1)[:16,:16,:]
				yield state, move[y][x], get_reward(nstate), nstate

def parse_file(file):
	data = json.load(open(file, "rt"))
	production = data["productions"]
	for t in range(data["num_frames"]-1):
		move = data["moves"][t]
		frame = data["frames"][t]
		nframe = data["frames"][t+1]
		for num in range(data["num_players"]):
			for s, a, r, ns in parse_sars(frame, move, nframe, production, num):
				yield s, a, r, ns

def generator(batch_size=32):
	files = glob(DATA_DIR + "**/*.hlt")
	random.shuffle(files)
	S, A, R, NS = [], [], [], []
	for file in files:
		for s, a, r, ns in parse_file(file):
			S.append(s)
			A.append(a)
			R.append(r)
			NS.append(ns)
			if len(S) == batch_size:
				yield list(map(np.array, [S, A, R, NS]))
				S, A, R, NS = [], [], [], []

if __name__ == "__main__":
	gen = generator()
	s, a, r, ns = next(gen)
	print(s.shape, a.shape, r.shape, ns.shape)
