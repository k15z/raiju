import json
import tqdm
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
	return reward

def get_ndarray(frame, production, num):
	width, height = len(frame[0]), len(frame)
	arr = np.zeros((height*2, width*2, 3))
	for y in range(height):
		for x in range(width):
			player, strength = frame[y][x]
			for dx in [1, 2]:
				for dy in [1, 2]:
					if player == num:
						arr[y*dy,x*dx,0] = 1.0
					elif player != 0:
						arr[y*dy,x*dx,0] = -1.0
					arr[y*dy,x*dx,1] = strength / 255.0
					arr[y*dy,x*dx,2] = production[y][x] / 255.0
	return arr

def parse_sarsa(frame, move, nframe, nmove, production, num):
	width, height = len(frame[0]), len(frame)
	arr = get_ndarray(frame, production, num)
	narr = get_ndarray(nframe, production, num)
	for y in range(height):
		for x in range(width):
			if frame[y][x][0] > 0.0:
				state = np.roll(np.roll(arr, -y+15, axis=0), -x+15, axis=1)[:30,:30,:]
				nstate = np.roll(np.roll(narr, -y+15, axis=0), -x+15, axis=1)[:30,:30,:]
				yield state, move[y][x], get_reward(nstate), nstate, nmove[y][x]

def parse_file(file):
	data = json.load(open(file, "rt"))
	production = data["productions"]
	for t in range(data["num_frames"]-2):
		move = data["moves"][t]
		nmove = data["moves"][t+1]
		frame = data["frames"][t]
		nframe = data["frames"][t+1]
		for num in range(1, data["num_players"]):
			for s, a, r, ns, na in parse_sarsa(frame, move, nframe, nmove, production, num):
				yield s, a, r, ns, na

def generator(batch_size=32):
	files = glob(DATA_DIR + "**/*.hlt")
	random.shuffle(files)
	S, A, R, NS, NA = [], [], [], [], []
	for file in files:
		for s, a, r, ns, na in parse_file(file):
			S.append(s)
			A.append(a)
			R.append(r)
			NS.append(ns)
			NA.append(na)
			if len(S) == batch_size:
				yield list(map(np.array, [S, A, R, NS, NA]))
				S, A, R, NS, NA = [], [], [], [], []

if __name__ == "__main__":
	gen = generator()
	i = 0
	for s, a, r, ns, na in tqdm.tqdm(gen):
		i += 1
	print(i)