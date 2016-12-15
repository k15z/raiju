import hlt
import random
import numpy as np
from model import restore, get_actions
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square

restore()
EPSILON = 0.8
MAX_SQUARES = 256

myID, game_map = hlt.get_init()
hlt.send_init("Raiju")

while True:
    game_map.get_frame()

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
                    elif square.owner != 0:
                        board[square.y*dy,square.x*dx,0] = -1.0
                    board[square.y*dy,square.x*dx,1] = square.strength / 255.0
                    board[square.y*dy,square.x*dx,2] = square.production / 255.0
    extras = []
    if len(squares) > MAX_SQUARES:
        random.shuffle(squares)
        extras = squares[MAX_SQUARES:]
        squares = squares[:MAX_SQUARES]

    actions = get_actions(np.array([
        np.roll(np.roll(board, -square.y+15, axis=0), -square.x+15, axis=1)[:30,:30,:] for square in squares
    ]))

    moves = []
    for i in range(len(squares)):
        square = squares[i]
        action = actions[i]
        if random.random() > EPSILON:
            action = random.choice((NORTH, EAST, SOUTH, WEST, STILL))
        if square.strength > 0:
            moves.append(Move(square, action))
    for extra_square in extras:
        action = random.choice((NORTH, EAST, SOUTH, WEST, STILL))
        if square.strength > 0:
            moves.append(Move(extra_square, action))
    hlt.send_frame(moves)
