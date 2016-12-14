import json
import random
import numpy as np
from os import path
from glob import glob

def get_reward(player, frame):
	reward = 0.0
	width, height = len(frame[0]), len(frame)
	for y in range(height):
		for x in range(width):
			p, s = frame[y][x]
			if p == player:
				reward += s
			elif p != 0:
				reward -= s
	return reward / (width * height)

def get_ndarray(player, frame, production):
	width, height = len(frame[0]), len(frame)
	arr = np.zeros((height*2, width*2, 3))
	for y in range(height):
		for x in range(width):
			p, strength = frame[y][x]
			for dx in [1, 2]:
				for dy in [1, 2]:
					arr[y*dy,x*dx,0] = 1.0 if player == p else -1.0
					arr[y*dy,x*dx,1] = strength / 255.0
					arr[y*dy,x*dx,2] = production[y][x] / 255.0
	return arr

def parse_frame(player, frame, move, production):
	width, height = len(frame[0]), len(frame)
	arr = get_ndarray(player, frame, production)
	for y in range(height):
		for x in range(width):
			if frame[y][x][0] > 0.0:
				state = np.roll(np.roll(arr, -y+15, axis=0), -x+15, axis=1)[:30,:30,:]
				yield state, move[y][x]

def parse_player(player, obj):
	moves = obj["moves"]
	frames = obj["frames"]
	production = obj["productions"]
	num_frames = obj["num_frames"]

	rewards = np.zeros([num_frames-1])
	for t in range(0, num_frames-1):
		rewards[t] = get_reward(player, frames[t+1])
	for t in range(0, num_frames-1):
		for u in range(t+1, num_frames-1):
			rewards[t] += rewards[u] * 0.99**(u-t)
	rewards = (rewards - np.mean(rewards)) / np.std(rewards)

	for t in range(0, num_frames-1):
		for state, action in parse_frame(player, frames[t], moves[t], production):
			yield state, action, rewards[t]

def load_file(filename):
	obj = json.load(open(filename, "rt"))
	for player in range(1, obj["num_players"]):
		for state, action, reward in parse_player(player, obj):
			yield state, action, reward

def generator(batch_size=32):
	files = glob("./data/*.hlt")
	random.shuffle(files)
	S, A, R = [], [], []
	for file in files:
		for s, a, r in load_file(file):
			S.append(s)
			A.append(a)
			R.append(r)
			if len(S) == batch_size:
				yield np.array(S), A, R
				S, A, R = [], [], []
