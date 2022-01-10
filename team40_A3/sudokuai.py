#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import math
import copy
import competitive_sudoku.sudokuai
import numpy as np
from collections import defaultdict
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
from typing import Any, Union, Tuple


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:

        def check_row_and_column_values(board: SudokuBoard, x, y):
            """
            @param board: A sudoku board.
            @param x: A row-coordinate.
            @param y: A column-coordinate.
            @Return: Two arrays, containing all values currently on the row and column.
            """
            column_values = []
            row_values = []
            for iterator in range(board.N):
                row_values.append(board.get(x, iterator))
                column_values.append(board.get(iterator, y))
            return row_values, column_values

        def get_all_empty_squares(board: SudokuBoard):
            """
            @param board: A sudoku board.
            @Return: an array with the coordinates of all the empty squares.
            """
            open_squares = [(i, j) for i in range(board.board_height()) for j in range(board.board_width()) if
                            board.get(i, j) == SudokuBoard.empty]
            return open_squares

        def check_sub_square_values(board: SudokuBoard, x, y):
            """
            @param board: A sudoku board.
            @param x: A row-coordinate.
            @param y: A column-coordinate.
            @Return: An array, containing all values of the sub-matrix that the given move is in.
            """
            sub_matrix_values = []
            (p, q) = (int(math.ceil((x + 1) / board.m) * board.m) - 1,
                      int(math.ceil(
                          (y + 1) / board.n) * board.n) - 1)  # calculates the highest coordinates in the sub-square
            (r, s) = (p - (board.m - 1), q - (board.n - 1))  # calculates the lowest coordinates in the sub-square
            for i in range(r, p + 1):
                for j in range(s, q + 1):
                    sub_matrix_values.append(board.get(i, j))
            return sub_matrix_values

        def possible(game_state: GameState):
            # TODO: Add the functionality that makes it so we only consider valid/legal moves. Not all possible moves.
            """
            @Param game_state: A GameState item.
            @Return: An array with all possible moves in the Move format (x-coord, y-coord, value).
            """
            all_moves = []  # this will contain all the moves in the end
            open_squares = get_all_empty_squares(game_state.board)
            for coords in open_squares:  # loop over all empty squares

                values_left = list(range(1,
                                         game_state.board.N + 1))  # This list wil eventually contain all the values possible on coordinate (i,j)
                row_values, column_values = check_row_and_column_values(game_state.board,
                                                                        *coords)  # Get all values on row and column for given move
                sub_square_values = check_sub_square_values(game_state.board,
                                                            *coords)  # Get all values in sub-matrix for given move
                joined_values = row_values + column_values + sub_square_values  # Put all values them together
                remaining_moves = [x for x in values_left if
                                   x not in joined_values]  # Keep only the values that are not on the board yet
                for value in remaining_moves:  # Add values to the list if not in tabooMove
                    if Move(coords[0], coords[1], value) not in game_state.taboo_moves:
                        all_moves.append(Move(coords[0], coords[1], value))

            return all_moves

        def score_eval(board: SudokuBoard, move: Move):
            """
            @param move: A move.
            @param board: A sudoku board.
            @Return: Return the value in points that the move has if it would be played on the board.
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

        def is_game_over(board: SudokuBoard):
            """
            @Param board: A sudoku board.
            @Return: A boolean signifying whether the board is full/game is over, or not.
            """
            game_over = False
            if len(get_all_empty_squares(board)) == 0:
                game_over = True
            return game_over

        def update_board(game_state: GameState, action):
            """
            @Param game_state: A GameState item.
            @Param action: A move that can be played on the board.
            @return: Returns a GameState item with the move played and the score updated.
            """
            # Update the board with a new move
            next_state = copy.deepcopy(game_state)
            next_state.board.put(action.i, action.j, action.value)
            calculated_score = score_eval(next_state.board, action)
            next_state.scores[next_state.current_player() - 1] += calculated_score
            return next_state

        def game_result(game_state: GameState, our_player_number):
            """
            @Param game_state: A GameState item.
            @Param our_player_number: An int signifying if we are player one or two.
            @Return: Returns 1 if we win on this board and -1 if we lose.
            """
            max_score = max(game_state.scores)
            if game_state.scores.index(max_score) == our_player_number:
                return 1
            else:
                return -1

        class MonteCarloTreeSearchNode:
            """
            These are the nodes of the tree!
            """

            def __init__(self, game_state: GameState, parent=None, parent_action=None, our_player_number=None):
                """
                @Param state: A GameState item.
                @Param parent: It is None for the root node and for other nodes it is equal to the node it is derived from.
                @Param children: It contains all child nodes for current node.
                @Param parent_action: None for the root node and for other nodes it is equal to the action which its parent carried out.
                @Param _number_of_visits: Number of times current node is visited
                @Param results: It’s a dictionary, will contain the score evaluation for the current node.
                @Param _untried_actions: Represents the list of all possible actions
                """
                self.state = game_state
                self.parent = parent
                self.parent_action = parent_action
                self.children = []
                self._number_of_visits = 0
                self._results = defaultdict(int)
                self._results[1] = 0
                self._results[-1] = 0
                self._untried_actions = self.untried_actions()
                self.our_player_number = our_player_number
                return

            def untried_actions(self):
                """
                @Return: Returns a list with all moves that are still possible from here on out.
                """
                self._untried_actions = possible(self.state)
                return self._untried_actions

            def score(self):
                """
                @Return: Returns the difference between wins and losses for the children nodes.
                """
                wins = self._results[1]
                loses = self._results[-1]
                return wins - loses

            def no_visits(self):
                """
                @Return: Returns the amount of time a node has been visited.
                """
                return self._number_of_visits

            def expand(self):
                """
                Expands the tree by selecting a random move to be played on the board and placing it.
                Returns the new child node.
                """
                action = random.choice(self._untried_actions)
                next_state = update_board(self.state, action)
                child_node = MonteCarloTreeSearchNode(next_state, parent=self, parent_action=action)
                self.children.append(child_node)

                action1 = random.choice(self._untried_actions)
                next_state1 = update_board(self.state, action1)
                child_node1 = MonteCarloTreeSearchNode(next_state1, parent=self, parent_action=action1)
                self.children.append(child_node1)
                return child_node

            def is_not_leaf_node(self):
                """
                Checks whether current node is leaf node
                """
                return len(self.children)

            def rollout(self):
                """
                This function goes all the way to the end of the game by filling in random (possible, not legal) moves and
                returns whether we win or lose on the final board.
                """
                current_rollout_state = self.state
                possible_moves = possible(current_rollout_state)
                while len(possible_moves) > 0:
                    action = self.rollout_policy(possible_moves)
                    current_rollout_state = update_board(current_rollout_state, action)
                    possible_moves = possible(current_rollout_state)
                    # if len(possible_moves) == 0:
                    #     break
                return game_result(current_rollout_state, self.our_player_number)

            def backpropagate(self, result):
                """
                This function iteratively loops from child to parent and updates the number of times visited.
                """
                self._number_of_visits += 1
                self._results[result] += 1
                if self.parent:
                    self.parent.backpropagate(result)

            def is_fully_expanded(self):
                """
                This function checks whether our current node has been fully checked.
                """
                return len(self._untried_actions) == 0

            def best_child(self, c_param=1.0):
                """
                This function chooses which child node is the best, based on the standard UCB formula in MCTS.
                """
                choices_weights = [
                    (c.score() / c.no_visits()) + c_param * np.sqrt((np.log(self.no_visits())) / c.no_visits()) for c in
                    self.children]
                return self.children[np.argmax(choices_weights)]

            def rollout_policy(self, possible_moves):
                """
                This function selects a child move to consider from the list of possible moves.
                We can edit this to change our strategy.
                """
                return possible_moves[np.random.randint(len(possible_moves))]

            def _tree_policy(self):
                """
                This function selects a node to perform the rollout on.
                Can also be edited to only consider certain nodes.
                """
                current_node = self
                while current_node.is_not_leaf_node():
                    current_node = current_node.best_child()

                if current_node.no_visits() != 0:
                    if not current_node.is_fully_expanded():
                        return current_node.expand()
                return current_node

        # make a copy of the game_state so we do not ruin the original.
        game_state2 = copy.deepcopy(game_state)
        # Initialize root of the tree
        root = MonteCarloTreeSearchNode(game_state2, our_player_number=game_state.current_player() - 1)
        # Perform rollout on a random basis.
        # selected_node = root.best_action()
        # Propose the best selected move.
        # self.propose_move(selected_node.parent_action)

        simulation_no = 1
        while True:
            v = root._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
            if simulation_no % 10 == 0:
                selected_node = root.best_child()
                self.propose_move(selected_node.parent_action)
            simulation_no += 1
