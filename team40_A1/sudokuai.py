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

    def compute_best_move(self, game_state: GameState) -> None:

        # Get the standard needed variables
        N = game_state.board.N  # depth of matrix
        n = game_state.board.n  # number of rows in a block
        m = game_state.board.m  # number of columns in a block
        player = 1              # This keeps track of whether we are player 1 or player 2



        open_squares_init = [(i, j) for i in range(N) for j in range(N) if game_state.initial_board.get(i, j) == SudokuBoard.empty]
        if len(game_state.moves) % 2 == 1:
            player = 2              # Change our player number to 2 if we are player two

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
            # Make a list of all the squares that have not yet been filled in
            open_squares = [(i, j) for i in range(N) for j in range(N) if
                            board.get(i, j) == SudokuBoard.empty]
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

        # This part of the code is unused and obsolete!!!!!!!!
        def evaluate_board(board: SudokuBoard):
            """
            @param board: A sudoku board.
            @Return: an integer with a numeric value. Higher = better board state
            """
            final_score = 1     # This value will be the final score for the board evaluation, all subroutines add or subtract from this score
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

        # This evaluation function uses the score of the board as the eventual evaluation function
        def score_eval(board: SudokuBoard, move: Move):
            """
             @param move: A move
             @param board: A sudoku board.
             @Return: Return the value in points that the move has if it would be played on the board
             """

            matrix = convert_to_matrix(board)               # Quick conversion to do simple calculations
            row_filled = column_filled = box_filled = True  # These variables are false when there is a 0 in their area on the board

            # This loop checks whether there is another 0 on the same row and column, if so switches state to false
            for iterator in range(N):
                if matrix[move.i][iterator] == 0:
                    row_filled = False
                if matrix[iterator][move.j] == 0:
                    column_filled = False

            # Calculate in which quadrant the given move falls.
            (p, q) = (int(math.ceil((move.i + 1) / n) * n) - 1, int(math.ceil((move.j + 1) / m) * m) - 1)  # calculates the highest coordinates in the sub-square
            (r, s) = (p - (n - 1), q - (m - 1))  # calculates the lowest coordinates in the sub-square

            # For the given quadrant, check whether any of the squares are filled with zero, if so switch to false
            for x in range(r, p + 1):
                for y in range(s, q + 1):
                    if board.get(x, y) == 0:
                        box_filled = False

            # Increase the score by the point value depending on how many areas inputting the move would close off
            boolean_list = [row_filled, column_filled, box_filled]
            true_values = sum(boolean_list)
            if true_values == 1:
                score = 1
            elif true_values == 2:
                score = 3
            elif true_values == 3:
                score = 7
            else:
                score = 0.01

            return score

        def minimax(board: SudokuBoard, depth, is_maximising_player, score_1, score_2):
            """
            @param board: A sudoku board.
            @param depth: The corresponding depth within the tree.
            @param is_maximising_player: True/False indicator for min/max search.
            @param score_1: Score calculation for player 1
            @param score_2: Score calculation for player 2
            @Return: return the best possible next move according to the minimax
            """

            all_moves_list = possible(board)  # Check all moves given a specific board

            if depth == max_depth or len(
                    all_moves_list) == 0:  # Checks whether we are in a leaf node or on the last possible move
                print("Final score: ", score_1 - score_2)  # This print is only here to show what score we calculated
                return score_1 - score_2

            if is_maximising_player:  # Check whether we are the maximising player
                value = -math.inf
                max_value = 0  # Temporary counter to compare against

                for move in all_moves_list:

                    score_1_temp = copy.deepcopy(score_1)  # Copy the score value
                    board.put(move.i, move.j, move.value)  # Actually play the move in the board
                    score_1 += score_eval(board, move)  # Increase the score by whatever the move would be worth
                    value = max(value, minimax(board, depth + 1, False, score_1, score_2))  # Dive into the recursive structure

                    board.put(move.i, move.j, 0)  # Revert the played move
                    score_1 = copy.deepcopy(score_1_temp)  # Revert the score to the value before we played the above move

                    if depth == 0 and value > max_value:  # if depth == 0, Update max_value and propose the move
                        max_value = value
                        self.propose_move(move)

                return value  # Return the value (Not sure if this is necessary)

            else:  # If we are not the maximizing player we end up here
                value = math.inf  # Declare highest possible number to compare negative against

                for move in all_moves_list:  # For all moves possible:

                    score_2_temp = score_2  # Copy score_2 over (Also not sure if necessary)
                    board2 = copy.deepcopy(board)  # TODO: Hier gebeurt dus nog iets fucky's. Als we achter staan in de laatste beurt kiest ie geen move om te doen omdat alle values negatief blijken te zijn (zie print statement)
                    board2.put(move.i, move.j, move.value)
                    score_2_temp += score_eval(board2, move)
                    value = min(value, minimax(board2, depth + 1, True, score_1, score_2_temp))

                return value




        def minimax_alpha_beta(board: SudokuBoard, depth, alpha, beta, is_maximising_player, score_1, score_2):
            """
            @param board: A sudoku board.
            @param depth: The corresponding depth within the tree.
            @param is_maximising_player: True/False indicator for min/max search.
            @param score_1: Score calculation for player 1
            @param score_2: Score calculation for player 2
            @Return: return the best possible next move according to the minimax
            """

            all_moves_list = possible(board)  # Check all moves given a specific board

            if depth == max_depth or len(
                    all_moves_list) == 0:  # Checks whether we are in a leaf node or on the last possible move
                print("Final score: ", score_1 - score_2)  # This print is only here to show what score we calculated
                return score_1 - score_2

            if is_maximising_player:  # Check whether we are the maximising player
                max_evaluation = -math.inf
                max_value = 0  # Temporary counter to compare against

                for move in all_moves_list:

                    score_1_temp = copy.deepcopy(score_1)  # Copy the score value
                    board.put(move.i, move.j, move.value)  # Actually play the move in the board
                    score_1 += score_eval(board, move)  # Increase the score by whatever the move would be worth
                    value = minimax_alpha_beta(board, depth + 1, alpha, beta, False, score_1, score_2) # Dive into the recursive structure
                    max_evaluation = max(value, max_evaluation)
                    alpha = max(alpha, value)

                    if beta <= alpha:
                        break

                    board.put(move.i, move.j, 0)  # Revert the played move
                    score_1 = copy.deepcopy(score_1_temp)  # Revert the score to the value before we played the above move

                    if depth == 0 and max_evaluation > max_value:  # if depth == 0, Update max_value and propose the move
                        max_value = max_evaluation
                        self.propose_move(move)

                return max_evaluation  # Return the value (Not sure if this is necessary)

            else:  # If we are not the maximizing player we end up here
                min_evaluation = math.inf  # Declare highest possible number to compare negative against

                for move in all_moves_list:  # For all moves possible:


                    score_2_temp = score_2  # Copy score_2 over (Also not sure if necessary)
                    board2 = copy.deepcopy(board)  # TODO: Hier gebeurt dus nog iets fucky's. Als we achter staan in de laatste beurt kiest ie geen move om te doen omdat alle values negatief blijken te zijn (zie print statement)
                    board2.put(move.i, move.j, move.value)
                    score_2_temp += score_eval(board2, move)

                    value = minimax_alpha_beta(board, depth + 1, alpha, beta, False, score_1, score_2)
                    min_evaluation = min(value, min_evaluation)
                    beta = min(beta, value)
                return min_evaluation


        max_depth = 0  # Initialize max_depth

        # This is the iterative deepening code, it's very crude but it could be improved (for now always start at 0)
        for i in range(0, 3):
            max_depth = i  # Update the max depth
            # minimax(game_state.board, 0, True, game_state.scores[0],
            #         game_state.scores[1])  # call the minmax function for the given max_depth

            minimax_alpha_beta(game_state.board, 0,-math.inf, math.inf, True, game_state.scores[0], game_state.scores[1])  # call the minmax function for the given max_depth

        # Uncomment om een illegale move te doen zodat je terminal niet vol wordt gespammed
        # self.propose_move(Move(3, 2, 3))