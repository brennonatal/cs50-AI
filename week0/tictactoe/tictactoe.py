"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None

def initial_state():
    # Returns starting state of the board.
    
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    # Returns player who has the next turn on a board.
    X_counter = 0
    O_counter = 0

    for row in board:
        for cell in row:
            if cell == X:
                X_counter += 1
            elif cell == O:
                O_counter += 1
    if X_counter == O_counter:
        return X
    else:
        return O

def actions(board):
    # Returns set of all possible actions (i, j) available on the board.  
 
    actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                actions.add((i, j))
    return actions

def result(board, action):
    # Returns the board that results from making move (i, j) on the board.
    board_copy = deepcopy(board)

    i, j = action[0], action[1]

    if board_copy[i][j] == EMPTY:
        board_copy[i][j] = player(board)
        return board_copy
    else:
        raise ValueError("invalid option")

def winner(board):
    # Returns the winner of the game, if there is one.

    # checking for winner in rows
    for row in board:
        first_cell = row[0]
        if row[1] == first_cell == row[2]:
            return first_cell

    # checking for winner in collumns
    for col in range(3):
        first_cell = board[0][col]
        if board[1][col] == first_cell == board[2][col]:
            return first_cell

    # checking for winner in diagonals
    first_cell = board[0][0]
    if board[1][1] == first_cell == board[2][2]:
        return first_cell

    first_cell = board[0][2]
    if board[1][1] == first_cell == board[2][0]:
        return first_cell

    # no winner
    return None

def terminal(board):
    # Returns True if game is over, False otherwise.

    there_is_a_winner = winner(board)
    
    if there_is_a_winner == X or there_is_a_winner == O:
        return True

    for row in board:
        for cell in row:
            if cell == EMPTY:
                return False
    
    return True
    

def utility(board):
    # Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
   
    w = winner(board)

    if w == X:
        return 1
    elif w == O:
        return -1
    return 0

def maxValue(board):
    v = float("-inf")

    if terminal(board):
        return utility(board)

    for action in actions(board):
        v = max(v, minValue(result(board, action)))

    return v

def minValue(board):
    v = float("+inf")

    if terminal(board):
        return utility(board)

    for action in actions(board):
        v = min(v, maxValue(result(board, action)))
        
    return v


def minimax(board):
    # Returns the optimal action for the current player on the board.

    if terminal(board):
        return None

    p = player(board)
    # best_move = (0, 0)

    if p == X:
        max_value = float("-inf")
        for action in actions(board):
            play_value = minValue(result(board, action))
            if play_value > max_value:
                max_value = play_value
                best_move = action
    else:
        min_value = float("inf")
        for action in actions(board):
            play_value = maxValue(result(board, action))
            if play_value < min_value:
                min_value = play_value
                best_move = action
    
    return best_move
