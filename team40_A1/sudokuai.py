#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import numpy as np
import competitive_sudoku.sudokuai
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove, print_board


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:

        # Get the standard needed variables
        N = game_state.board.N          # depth of matrix
        n = game_state.board.n          # number of rows in a block
        m = game_state.board.m          # number of columns in a block

        # Make a list of all the squares that have not yet been filled in
        open_squares = [(i,j) for i in range(N) for j in range(N) if game_state.board.get(i, j) == SudokuBoard.empty]

        def convert_to_matrix(board: SudokuBoard):
            """
            @param board: A sudoku board.
            @Return: a 2 array of the given board
            """
            matrix = np.reshape(np.array(board.squares), (N, N))
            return matrix

        def possible(board: SudokuBoard):
            """
            @param board: A sudoku board.
            @Return: an array with all possible/legal moves in the Move format (x-coord, y-coord, value)
            """
            matrix = convert_to_matrix(board)
            all_moves = []              # this will contain all the moves in the end
            for coords in open_squares:  # loop over all empty squares

                # calculate sub-squares and prepare list of possible values
                possible_values = list(range(1, N+1))        # This list wil eventually contain all the values possible on coordinate (i,j)
                (p, q) = (np.int(np.ceil((coords[0] + 1) / n) * n)-1, np.int(np.ceil((coords[1] + 1) / m) * m)-1)   # calculates the highest coordinates in the sub-square
                (r, s) = (p-(n-1),q-(m-1))                                                          # calculates the lowest coordinates in the sub-square

                # makes a list of all values in row/column and box
                row_vals = np.unique(matrix[coords[0], :])
                col_vals = np.unique(matrix[:, coords[1]])
                box_vals = np.unique(matrix[r:p, s:q])
                all_values = np.concatenate((row_vals, col_vals, box_vals), axis=None)

                # remove all values in row/column/box from the list of possible values and return it
                values_left = [x for x in possible_values if x not in np.unique(all_values)]
                for x in range(len(values_left)):
                    if TabooMove(coords[0], coords[1], values_left[x]) in game_state.taboo_moves:
                        continue
                    all_moves.append(Move(coords[0], coords[1], values_left[x]))

            return all_moves

        def evaluate_board(board: SudokuBoard):
            """
            @param board: A sudoku board.
            @Return: an integer with a numeric value. Higher = better board state
            """
            final_score = 0     # This value will be the final score for the board evaluation, all subroutines add or subtract from this score
            matrix = convert_to_matrix(board)

            # These loops increase the evaluation score for each row/column that has one place left to fill in (it can increase our score)
            for i in range(N):
                row_counter = 0
                column_counter = 0
                for j in range(N):
                    if matrix[i][j] == 0:
                        row_counter = row_counter+1
                    if matrix[j][i] == 0:
                        column_counter = column_counter+1
                if row_counter == 1:
                    final_score = final_score+1
                if column_counter == 1:
                    final_score = final_score+1

            # calculate box_scores
            # create all the sub_squares
            sub_squares = [[matrix[j][i] for j in range(x, x + m) for i in range(y, y + n)] for x in range(0, N, m)for y in range(0, N, n)]

            # Checks if there is only one zero in a sub-square and increases the counter if true
            for i in range(len(sub_squares)):

                if sub_squares[i].count(0) == 1:
                    final_score = final_score+1

            return final_score

        evaluation_score = evaluate_board(game_state.board)

        # This is how we eventually submit our proposed move
        final_move_list = possible(game_state.board)
        move = random.choice(final_move_list)
        self.propose_move(move)

        while True:
            time.sleep(0.2)
            self.propose_move(random.choice(final_move_list))
