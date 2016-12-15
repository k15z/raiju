import json
import random
import numpy as np
from os import path
from glob import glob
from tqdm import tqdm

DATA_DIR = path.dirname(path.abspath(__file__))

def get_reward(state):
	reward = 0.0
	for y in range(state.shape[0]):
		for x in range(state.shape[1]):
			reward += state[y,x,0]# * state[y,x,1]
	return reward

def get_ndarray(frame, production, player_num):
	width, height = len(frame[0]), len(frame)
	arr = np.zeros((height*2, width*2, 3))
	for y in range(height):
		for x in range(width):
			player, strength = frame[y][x]
			for dx in [1, 2]:
				for dy in [1, 2]:
					if player == player_num:
						arr[y*dy,x*dx,0] = 1.0
					elif player != 0:
						arr[y*dy,x*dx,0] = -1.0
					arr[y*dy,x*dx,1] = strength / 255.0
					arr[y*dy,x*dx,2] = production[y][x] / 255.0
	return arr

def parse_sars(frame, move, nframe, production, player_num):
	width, height = len(frame[0]), len(frame)
	arr = get_ndarray(frame, production, player_num)
	narr = get_ndarray(nframe, production, player_num)
	for y in range(height):
		for x in range(width):
			if frame[y][x][0] > 0.0:
				state = np.roll(np.roll(arr, -y+15, axis=0), -x+15, axis=1)[:30,:30,:]
				nstate = np.roll(np.roll(narr, -y+15, axis=0), -x+15, axis=1)[:30,:30,:]
				yield state, move[y][x], get_reward(nstate), nstate

def parse_file(file):
	data = json.load(open(file, "rt"))
	production = data["productions"]
	times = list(range(data["num_frames"]-1))
	random.shuffle(times)
	for t in times:
		move = data["moves"][t]
		frame = data["frames"][t]
		nframe = data["frames"][t+1]
		for player_num in range(1, data["num_players"]):
			for s, a, r, ns in parse_sars(frame, move, nframe, production, player_num):
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
	for s, a, r, ns in tqdm(generator(batch_size=1)):
		pass
