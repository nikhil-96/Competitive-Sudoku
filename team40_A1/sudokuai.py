#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import copy
import math
import array
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
        N = game_state.board.N  # depth of matrix
        n = game_state.board.n  # number of rows in a block
        m = game_state.board.m  # number of columns in a block

        # Make a list of all the squares that have not yet been filled in
        open_squares = [(i, j) for i in range(N) for j in range(N) if game_state.board.get(i, j) == SudokuBoard.empty]

        def convert_to_matrix(board: SudokuBoard):
            """
            @param board: A sudoku board.
            @Return: a 2D array of the given board
            """
            matrix = [board.squares[i:i+N] for i in range(0, len(board.squares), N)]
            return matrix

        def possible(board: SudokuBoard):
            """
            @param board: A sudoku board.
            @Return: an array with all possible/legal moves in the Move format (x-coord, y-coord, value)
            """

            all_moves = []              # this will contain all the moves in the end
            for coords in open_squares:  # loop over all empty squares

                # calculate sub-squares and prepare list of possible values
                values_left = list(range(1, N+1))        # This list wil eventually contain all the values possible on coordinate (i,j)
                (p, q) = (int(math.ceil((coords[0] + 1) / n) * n)-1, int(math.ceil((coords[1] + 1) / m) * m)-1)   # calculates the highest coordinates in the sub-square
                (r, s) = (p-(n-1), q-(m-1))                                                          # calculates the lowest coordinates in the sub-square

                # remove all values that already exist on the same row/column/box as coords from the possible value list for that coord.
                for i in range(N):
                    if board.get(coords[0], i) in values_left:
                        values_left.remove(board.get(coords[0], i))
                    if board.get(i, coords[1]) in values_left:
                        values_left.remove(board.get(i, coords[1]))
                for x in range(r, p+1):
                    for y in range(s, q+1):
                        if board.get(x, y) in values_left:
                            values_left.remove(board.get(x, y))

                for value in values_left:
                    all_moves.append(Move(coords[0], coords[1], value))

            # We input all moves and then check the oracle to see which ones are illegal
            for move in all_moves:
                if move in game_state.taboo_moves:
                    all_moves.remove(move)

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

        def minimax(board: SudokuBoard, depth, isMaximisingPlayer, real_board=game_state.board):
            """
            @param board: A sudoku board.
            @param depth: The corresponding depth within the tree.
            @param isMaximisingPlayer: True/False indicator for min/max search.
            @Return: return the best possible next move according to the minimax
            ONLY WORKS FOR DEPTH == 2 YET
            """

            if depth == 2:
                return evaluate_board(board)  # returned value: the board after the 'best' move from opponent.

            all_moves_list = possible(board)

            if isMaximisingPlayer:
                value = -math.inf
                max_value = 0
                for move in all_moves_list:                                     # go over all possible moves our player can play
                    board.put(move.i, move.j, move.value)                       # actually play the move in the board
                    value = max(value, minimax(board, depth+1, False, board))   # recursive call to find all values after analyzing all possibles moves from opponent (when depth==2 is used)
                                                                                # the last board in the function call is the board with the move implented, so in else statement this board can be reset.
                    if depth == 0 and value > max_value:                        # you only come here when depth ==0.
                        max_value = value
                        self.propose_move(move)

                    board = copy.deepcopy(game_state.board)    # reset the board to the one the game behaves now, in order to analyze the effects of the next possible move

                return value                                   # the returned value is now the best value for you, after analyzing the opponents best responses to your moves

            else:
                value = +math.inf
                for move in all_moves_list:
                    board2 = copy.deepcopy(real_board)
                    board2.put(move.i, move.j, move.value)
                    value = min(value, minimax(board2, depth + 1, True))
                return value

        def minimax_alpha_beta(board: SudokuBoard, depth, alpha, beta, isMaximisingPlayer, real_board=game_state.board):
            """
            @param board: A sudoku board.
            @param depth: The corresponding depth within the tree.
            @param isMaximisingPlayer: True/False indicator for min/max search.
            @Return: return the best possible next move according to the minimax
            ONLY WORKS FOR DEPTH == 2 YET
            """

            if depth == 2:
                return evaluate_board(board)  # returned value: the board after the 'best' move from opponent.

            all_moves_list = possible(board)

            if isMaximisingPlayer:
                max_value = -math.inf
                if depth == 0:
                    value_max = 0
                for move in all_moves_list:
                    board.put(move.i, move.j, move.value)
                    value = minimax_alpha_beta(board, depth + 1, alpha, beta, False, board)
                    max_value = max(max_value, value)

                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break                 #only works when we searh deeper
                                              #so when we're more often in is isMaximisingPlayer during the recursive loops

                    if depth == 0 and value > value_max:
                        max_value = value
                        self.propose_move(move)

                    board = copy.deepcopy(game_state.board)

                return value

            else:
                min_value = +math.inf
                for move in all_moves_list:
                    board2 = copy.deepcopy(real_board)
                    board2.put(move.i, move.j, move.value)
                    value = minimax_alpha_beta(board2, depth + 1, alpha, beta, False)
                    min_value = min(min_value, value)
                return min_value

        # minimax(game_state.board, 0, True)
        minimax_alpha_beta(game_state.board, 0, -math.inf, math.inf, True)