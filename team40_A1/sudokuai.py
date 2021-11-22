#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import numpy as np
import competitive_sudoku.sudokuai
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        #Get the standard needed variables
        N = game_state.board.N
        n = game_state.board.n
        m = game_state.board.m

        #make a 2d matrix representation
        matrix = np.reshape(np.array(game_state.board.squares), (N,N))

        #Make a list of all the squares that have not yet been filled in
        open_squares = [(i,j) for i in range(N) for j in range(N) if game_state.board.get(i, j) == SudokuBoard.empty]

        #check which values are allowed for the given move, return a list of all possible values for that coordinate
        def possible(i, j):

            #caluclate subsquares and prepare list of possible values
            possible_values = list(range(1, N+1))        # This list wil eventually contain all the values possible on coördinate (i,j)
            (p, q) = (np.int(np.ceil((i + 1) / n) * n)-1, np.int(np.ceil((j + 1) / m) * m)-1)   #calculates the lowest coordinates in sub-square
            (r, s) = (p-(n-1),q-(m-1))                                                          #calculates the highest coordinates in the sub-square

            #make list of all values in row/column and box
            row_vals = np.unique(matrix[i,:])
            col_vals = np.unique(matrix[:,j])
            box_vals = np.unique(matrix[r:p, s:q])
            all_values = np.concatenate((row_vals, col_vals, box_vals), axis=None)

            #remove all values in row/column/box from the list of possible values and return it
            values_left = [x for x in possible_values if x not in np.unique(all_values)]
            return values_left


        #This loop concatenates all open coordinate squares with all possible values that it can fill in there
        all_moves = []
        for coords in open_squares:
            possibilities_list = possible(coords[0], coords[1])

            for x in range(len(possibilities_list)):
                if TabooMove(coords[0], coords[1], possibilities_list[x]) in game_state.taboo_moves:
                    continue
                all_moves.append(Move(coords[0], coords[1], possibilities_list[x]))

        print("These are all the possible moves: ")
        for item in all_moves:
            print(item)

        print("amount of taboo_moves: ", len(game_state.taboo_moves))

        move = random.choice(all_moves)
        self.propose_move(move)
        while True:
            time.sleep(0.2)
            self.propose_move(random.choice(all_moves))
