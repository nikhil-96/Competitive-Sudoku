#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)
import fnmatch
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

        self.top_move = Move(0, 0, 0)
        self.max_value = 0
        self.max_value_start = 0

    def compute_best_move(self, game_state: GameState) -> None:

        def check_if_first_player():
            """
            @Return: True if we are player 1 and False if we are player 2
            """
            if len(game_state.moves) % 2 == 1:
                return False
            else:
                return True

        def get_all_empty_squares(board: SudokuBoard):
            """
            @param board: A sudoku board.
            @Return: an array with the coordinates of all the empty squares
            """
            open_squares = [(i, j) for i in range(board.N) for j in range(board.N) if board.get(i, j) == SudokuBoard.empty]
            return open_squares

        def check_row_and_column_values(board: SudokuBoard, x, y):
            """
            @param board: A sudoku board.
            @param x: A row-coordinate
            @param y: A column-coordinate
            @Return: Two arrays, containing all values currently on the row and column
            """
            column_values = []
            row_values = []
            for iterator in range(board.N):
                row_values.append(board.get(x, iterator))
                column_values.append(board.get(iterator, y))
            return row_values, column_values

        def check_sub_square_values(board: SudokuBoard, x, y):
            """
            @param board: A sudoku board.
            @param x: A row-coordinate
            @param y: A column-coordinate
            @Return: An array, containing all values of the sub-matrix that the given move is in
            """
            sub_matrix_values = []
            (p, q) = (int(math.ceil((x + 1) / board.m) * board.m) - 1, int(math.ceil((y + 1) / board.n) * board.n) - 1)  # calculates the highest coordinates in the sub-square
            (r, s) = (p - (board.m - 1), q - (board.n - 1))  # calculates the lowest coordinates in the sub-square
            for i in range(r, p + 1):
                for j in range(s, q + 1):
                    sub_matrix_values.append(board.get(i, j))
            return sub_matrix_values

        def possible(board: SudokuBoard):
            """
            @param board: A sudoku board.
            @Return: an array with all possible/legal moves in the Move format (x-coord, y-coord, value)
            """
            all_moves = []              # this will contain all the moves in the end
            open_squares = get_all_empty_squares(board)
            for coords in open_squares:  # loop over all empty squares

                values_left = list(range(1, board.N+1))                                           # This list wil eventually contain all the values possible on coordinate (i,j)
                row_values, column_values = check_row_and_column_values(board, *coords)     # Get all values on row and column for given move
                sub_square_values = check_sub_square_values(board, *coords)                 # Get all values in sub-matrix for given move
                joined_values = row_values + column_values + sub_square_values              # Put all values them together
                remaining_moves = [x for x in values_left if x not in joined_values]        # Keep only the values that are not on the board yet

                for value in remaining_moves:                                               # Add values to the list if not in tabooMove
                    if Move(coords[0], coords[1], value) not in game_state.taboo_moves:
                        all_moves.append(Move(coords[0], coords[1], value))

            return all_moves

        # This evaluation function uses the score of the board as the eventual evaluation function
        def score_eval(board: SudokuBoard, move: Move):
            """
             @param move: A move
             @param board: A sudoku board.
             @Return: Return the value in points that the move has if it would be played on the board
             """
            row_filled = column_filled = box_filled = True  # These variables are false when there is a 0 in their area on the board

            row_values, column_values = check_row_and_column_values(board, move.i, move.j)
            sub_square_values = check_sub_square_values(board, move.i, move.j)

            if 0 in row_values:
                row_filled = False
            if 0 in column_values:
                column_filled = False
            if 0 in sub_square_values:
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
                score = 0
            return score

        def minimax_alpha_beta(board: SudokuBoard, depth, alpha, beta, is_maximising_player):
            """
            @param board: A current sudoku board.
            @param depth: The corresponding depth within the tree.
            @param is_maximising_player: True/False indicator for min/max search.
            @Return: return the best possible next move according to the minimax
            """

            all_moves_list = possible(game_state2.board)                      # Check all moves on the copied board

            if depth == max_depth or len(all_moves_list) == 0:                # Checks whether we are in a leaf node or on the last possible move
                return game_state2.scores[0]-game_state2.scores[1]

            if is_maximising_player:                                          # Check whether we are the maximising player
                max_evaluation = -math.inf

                for move in all_moves_list:

                    # This chunk places the move on a copy of the board, evaluates it and updates the copied score
                    game_state2.board.put(move.i, move.j, move.value)
                    calculated_score = score_eval(game_state2.board, move)
                    game_state2.scores[0] += calculated_score

                    value = minimax_alpha_beta(game_state2.board, depth + 1, alpha, beta, False)    # Here we go into recursion
                    max_evaluation = max(value, max_evaluation)

                    # After the recursion we remove the move and also re-calculate the score
                    game_state2.board.put(move.i, move.j, 0)
                    game_state2.scores[0] = game_state2.scores[0]-calculated_score

                    alpha = max(alpha, value)

                    if beta <= alpha:
                        break

                    if depth == 0 and move not in game_state.taboo_moves and max_evaluation > self.max_value_start:          # if depth == 0 and also not a taboo_move, propose it
                        if max_evaluation > self.max_value:
                            self.max_value = max_evaluation
                            self.top_move = Move(move.i, move.j, move.value)
                    elif depth == 0 and move not in game_state.taboo_moves and max_evaluation == self.max_value == self.max_value_start:
                        self.top_move = Move(move.i, move.j, move.value)

                return max_evaluation                             # Return the value (Not sure if this is necessary)

            else:                                                 # If we are not the maximizing player we end up here
                min_evaluation = math.inf                         # Declare highest possible number to compare negative against
                for move in all_moves_list:                       # iterate over all the enemies moves

                    # Once again, place the move on the board and update the score
                    game_state2.board.put(move.i, move.j, move.value)
                    calculated_score2 = score_eval(game_state2.board, move)
                    game_state2.scores[1] += calculated_score2

                    value = minimax_alpha_beta(game_state2.board, alpha, beta, depth + 1, True)  # Another recursive loop
                    min_evaluation = max(value, min_evaluation)

                    # Revert the played move and revert the scores to how it was
                    game_state2.board.put(move.i, move.j, 0)
                    game_state2.scores[1] = game_state2.scores[1] - calculated_score2

                    beta = min(beta, value)
                    if beta <= alpha:
                        break

                return min_evaluation

        all_possible_moves = possible(game_state.board)
        game_state2 = copy.deepcopy(game_state)

        # Here we will select appropriate strategy
        too_many_moves = 30
        move_chosen = False
        # Case: too many moves->random non_committal_move
        if len(all_possible_moves) > too_many_moves:
            open_squares = get_all_empty_squares(game_state2.board)
            print("Length of possible moves = ", len(all_possible_moves))
            for moves in all_possible_moves:  # loop over all empty squares
                row_values, column_values = check_row_and_column_values(game_state2.board, moves.i, moves.j)
                sub_square_values = check_sub_square_values(game_state2.board, moves.i, moves.j)
                print("Move: ", moves, " Row: ", row_values, " Column: ", column_values, " Matrix values: ", sub_square_values)
                if row_values.count(0) == 1 or column_values.count(0) == 1 or sub_square_values.count(0) == 1:
                    self.propose_move(moves)
                    move_chosen = True

            while not move_chosen:
                considered_move = random.choice(all_possible_moves)
                game_state2.board.put(considered_move.i, considered_move.j, considered_move.value)
                if score_eval(game_state2.board, considered_move) != 0 and considered_move not in game_state.taboo_moves:
                    self.propose_move(considered_move)
                    game_state2.board.put(considered_move.i, considered_move.j, 0)
                    break

        # Case: Minimax
        else:
            self.max_value = 0
            # Swap the scores if we are the other player
            if not check_if_first_player():
                game_state2.scores[0], game_state2.scores[1] = game_state2.scores[1], game_state2.scores[0]
                self.max_value = game_state2.scores[0] - game_state2.scores[1]
                self.max_value_start = game_state2.scores[0] - game_state2.scores[1]

            for max_depth in range(0, 15):                                              # Update the max depth
                minimax_alpha_beta(game_state2.board, 0, -math.inf, math.inf, True)      # call the minmax function for the given max_depth
                self.propose_move(self.top_move)


