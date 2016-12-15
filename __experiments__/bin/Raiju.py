import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
import random
import random
import numpy as np
from raiju_model import get_actions

myID, game_map = hlt.get_init()
hlt.send_init("Raiju")

curr_time = 0
while True:
    curr_time += 1
    game_map.get_frame()
    GAME_LENGTH = 10.0 * (game_map.height * game_map.width) ** 0.5

    squares = []
    board = np.zeros((game_map.height*2, game_map.width*2, 3))
    for y in range(game_map.height):
        for x in range(game_map.width):
            square = game_map.contents[y][x]
            for dx in [1, 2]:
                for dy in [1, 2]:
                    if square.owner == myID:
                        squares.append(square)
                        board[square.y*dy,square.x*dx,0] = 1.0
                    else: board[square.y*dy,square.x*dx,0] = -1.0
                    board[square.y*dy,square.x*dx,1] = square.strength / 255.0
                    board[square.y*dy,square.x*dx,2] = square.production / 255.0
    MAX_SQUARES = 256
    if len(squares) > MAX_SQUARES:
        random.shuffle(squares)
        squares = squares[:MAX_SQUARES]

    states = []
    times = []
    for square in squares:
        state = np.roll(np.roll(board, -square.y+18, axis=0), -square.x+18, axis=1)[:36,:36,:]
        states.append(state)
        times.append([curr_time / GAME_LENGTH])
    states = np.array(states)
    times= np.array(times)
    actions = get_actions(states, times)

    moves = []
    for i in range(len(states)):
        square = squares[i]
        action = actions[i]
        if random.random() > 0.3 and square.strength > 0:
            moves.append(Move(square, action))
        else:
            if square.strength > 0:
                action = random.choice((NORTH, EAST, SOUTH, WEST, STILL))
                moves.append(Move(square, action))
    hlt.send_frame(moves)
