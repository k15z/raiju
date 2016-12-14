import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
import random

myID, game_map = hlt.get_init()
hlt.send_init("ZBot2")

while True:
    game_map.get_frame()
    for y in range(game_map.height):
	    for x in range(game_map.width):
	    	x, y, owner, strength, production = game_map.contents[y][x]
    moves = [Move(square, random.choice((NORTH, EAST, SOUTH, WEST, STILL))) for square in game_map if square.owner == myID]
    hlt.send_frame(moves)
