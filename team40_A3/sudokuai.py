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

        def possible_allmoves(game_state: GameState):
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

        def possible(game_state: GameState):
            """
           @Param game_state: A GameState item.
           @Return: An array with all possible moves in the Move format (x-coord, y-coord, value).
           """
            taboo_moves = get_taboo_moves(game_state)
            all_moves = possible_allmoves(game_state)

            return_moves = []
            for item in all_moves:
                if item not in taboo_moves:
                    return_moves.append(item)
            return return_moves

        def get_taboo_moves(game_state):
            """
            @Return: an array with some taboo_moves based on rows columns and submatrices,
            format is [(x-coord, y-coord), value]
            """
            all_moves = possible_allmoves(game_state)  # Get all possible moves
            single_value_coordinates = []  # Initialize the list of moves with only one value option
            selected_items = [tuple([item.i, item.j]) for item in
                              all_moves]  # Fill a list with only the coordinates, not the value
            taboo_moves = []  # Initialize the return list

            # Loop over all possible moves and extract the ones that only have one value option
            for a in all_moves:
                if selected_items.count(tuple([a.i, a.j])) == 1:
                    single_value_coordinates.append(a)

            # Now check whether any other empty square on the same row, column or sub-square has the same value option,
            # that is then a taboo_move
            for single_moves in single_value_coordinates:
                for move in all_moves:
                    if single_moves != move and single_moves.value == move.value and \
                            (single_moves.j == move.j or single_moves.i == move.i or
                             check_sub_square_values(game_state.board, single_moves.i, single_moves.j) ==
                             check_sub_square_values(game_state.board, move.i, move.j)):
                        taboo_moves.append(move)
            return taboo_moves

        def find_best_candidate_move(dict_of_moves: dict[int: tuple]) -> tuple:
            """ Returns a tuple with coordinates of one of the best moves from the given move dictionary

            :param dict_of_moves: dictionary with scores as index and move tuples as values
            :return: tuple of one of the best scores in the given dictionary
            """
            # find the highest scoring move possible from the given dict
            for index in range(3, -1, -1):
                if dict_of_moves[f"score{index}"]:
                    return random.choice(dict_of_moves[f"score{index}"])

        def get_random_set_element(set_x: set) -> Any:
            """ Given a set, return a random element from that set without changing the set
            with the fastest Python implementation.

            :param set_x: Given set to return a random element of
            :return: Random element of set_x
            """
            elem = set_x.pop()
            set_x.add(elem)

            return elem

        def does_move_complete_row(
                state: GameState,
                i: int,
                j: int,
        ) -> Tuple[bool, set]:
            """ Checks whether a move to the given field could complete a row
            and also returns illegal numbers in that row

            :param state: object of the current state of the game
            :param i: row index of the move
            :param j: column index of the move
            :return: Tuple of the bool value representing
             whether filling the cell gives the player a point for that row.
            """
            illegal_numbers_set = {0}
            is_row_filled = True

            # loop over the different columns of the row of the given cell,
            # and check if there are any other empty cells.
            for col_index in range(state.board.N):

                # Skip the current cell (We already know it's empty)
                if j == col_index:
                    continue

                # Add the value of the cells
                # in that row to the set of illegal numbers.
                cell = state.board.get(i, col_index)
                # Heuristic 1 - remove already played numbers in row from legal moves
                illegal_numbers_set.add(cell)

                # If a cell is empty, then filling another empty cell
                # will not result in filling the row.
                if cell == state.board.empty:
                    is_row_filled = False

            return is_row_filled, illegal_numbers_set

        def does_move_complete_subgrid(
                state: GameState,
                i: int,
                j: int,
        ) -> Tuple[bool, set]:
            """ Checks whether a move to the given field could complete
             a subgrid and returns found illegal numbers in that subgrid

            :param state: object of the current state of the game
            :param i: row index of the move
            :param j: column index of the move
            :return: tuple of bool value representing whether filling
            the cell gives the player a point for that subgrid and the set of
            illegal (taken) numbers from that subgrid
            """
            # boolean that represents if filling the cell
            # gives the player a point for that subgrid.
            is_subgrid_filled = True
            illegal_numbers_set = {0}

            # Compute the starting and ending row
            # and column indices for the subgrid of the cell.
            row_subgrid_start = (i // state.board.m) * state.board.m
            row_subgrid_end = ((i // state.board.m) + 1) * state.board.m
            col_subgrid_start = (j // state.board.n) * state.board.n
            col_subgrid_end = ((j // state.board.n) + 1) * state.board.n

            # loop over the cells of the subgrid
            # and check if there are any other empty cells.
            for subgrid_row_index in range(row_subgrid_start, row_subgrid_end):

                for subgrid_col_index in range(col_subgrid_start,
                                               col_subgrid_end):

                    # Skip the current cell (We already know it's empty)
                    if i == subgrid_row_index and j == subgrid_col_index:
                        continue

                    # Add the numbers in the cell's subgrid
                    # to the illegal_numbers set
                    cell = state.board.get(
                        subgrid_row_index, subgrid_col_index)
                    # Heuristic 1 - remove already played numbers in subgrid from legal moves
                    illegal_numbers_set.add(cell)

                    # If a cell in the subgrid is empty,
                    # then filling the given cell
                    # will result in no points for that subgrid
                    if cell == state.board.empty:
                        is_subgrid_filled = False

            return is_subgrid_filled, illegal_numbers_set

        def does_move_complete_column(
                state: GameState,
                i: int,
                j: int,
        ) -> Tuple[bool, set]:
            """ Checks whether a move to the given field could complete a column
            and updates the illegal numbers set on the column

            :param state: object of the current state of the game
            :param i: row index of the move
            :param j: column index of the move
            :return: tuple of bool value representing whether filling the cell
             gives the player a point for that column and the set of illegal
             numbers from that column
            """
            illegal_numbers_set = {0}
            is_col_filled = True
            # loop over the different rows of the column of the given cell,
            # and check if there are any other empty cells.
            for row_index in range(state.board.N):

                # Skip the current cell (We already know it's empty)
                if i == row_index:
                    continue

                # Add the value of the cells
                # in that column to the set of illegal numbers.
                cell = state.board.get(row_index, j)
                # Heuristic 1 - remove already played numbers in column from legal moves
                illegal_numbers_set.add(cell)

                # If a cell is empty, then filling another empty cell
                # will not result in filling the column.
                if cell == state.board.empty:
                    is_col_filled = False

            return is_col_filled, illegal_numbers_set

        def get_legal_move_list_from_dict(legal_move_dict: dict) -> list:
            """ Return a sorted-by-score list of moves from the given dict of legal moves with scores

            :param legal_move_dict: dictionary with scores as index and move tuples as values
            :return:
            """
            move_list = []
            # find the highest scoring move possible from the given dict
            # and add it to the list
            for index in range(3, -1, -1):
                if legal_move_dict[f"score{index}"]:
                    for move in legal_move_dict[f"score{index}"]:
                        move_list.append(move)

            return move_list

        def get_legal_moves(state: GameState, return_type: str) \
                -> Union[dict, list]:
            """ Returns list or dictionary of legal moves,
            depending on the requested return_type

            :param state: state of the game for which the moves are requested
            :param return_type: 'dict' or 'list' corresponding
             to the result type needed
            :return: Either a dictionary with scores as keys and corresponding
             moves as values if dict is requested,
             or simple list of legal moves otherwise
            """
            N = state.board.N

            # Initializing the dictionary that will contain the best moves.
            # Each of the scores shown below will contain
            # a list of tuples of the form (i ,j , value, score)
            # The 'i' and 'j' are the row and column indices of the cell.
            # Value is the number the cell is to be filled with.
            # Score refers to how good a move is right now -in a greedy manner.
            # score 0 means that playing the given move will result in 0 points
            # score 1 means that playing the given move will result in 1 point
            # score 2 means that playing the given move will result in 2 points
            # score 3 means that playing the given move will result in 7 points

            best_move_dict = {
                'score0': [],
                'score1': [],
                'score2': [],
                'score3': [],
            }

            # The set of all possible numbers given an empty board.
            possible_numbers = {
                number for number in range(state.board.N + 1)}

            # Loop over all cells in the board, find empty cells and
            # compute cell_info for each one.
            for i in range(N):
                for j in range(N):
                    cell = state.board.get(i, j)
                    if cell == state.board.empty:

                        # Set of numbers that we CAN'T use for this empty cell.
                        illegal_numbers = {0}
                        elem = None

                        # check whether filling this cell will
                        # complete a column, row, or subgrid
                        # note: We pass the illegal_numbers set by reference
                        # so we can compute the numbers already played
                        # in a column, row and subgrid,
                        # which saves an extra iteration later
                        will_fill_column, illegal_numbers_col = \
                            does_move_complete_column(state, i, j)
                        will_fill_row, illegal_numbers_row = \
                            does_move_complete_row(state, i, j)
                        will_fill_subgrid, illegal_numbers_subgrid = \
                            does_move_complete_subgrid(state, i, j)

                        illegal_numbers = illegal_numbers.union(
                            illegal_numbers_col,
                            illegal_numbers_row,
                            illegal_numbers_subgrid
                        )

                        # Heuristic 2 - assign score to every legal move
                        score = will_fill_column + will_fill_row + will_fill_subgrid

                        # The set of legal numbers for a given cell, is
                        # the set of all possible moves - set of illegal moves
                        legal_numbers = possible_numbers.difference(illegal_numbers)

                        try:
                            elem = get_random_set_element(legal_numbers)
                        except KeyError:
                            continue

                        # add the move to the best move list
                        # (if it's not a TabooMove)
                        if not (TabooMove(i, j, elem) in state.taboo_moves):
                            best_move_dict[f"score{score}"].append(
                                (i, j, elem, score)
                            )

            if return_type == "dict":
                return best_move_dict
            elif return_type == "list":
                return get_legal_move_list_from_dict(best_move_dict)

        def compute_greediest_move(state: GameState) -> tuple:
            """ Computes best move according to the greedy player's strategy

            :param state: current state of the game
            :return: tuple of the legal moves (with scores as keys)
             and the best move according to the greedy strategy
            """
            best_move_dict = get_legal_moves(state, "dict")

            return best_move_dict, find_best_candidate_move(best_move_dict)

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
                @Param children: It contains all possible actions from the current node.
                @Param parent_action: None for the root node and for other nodes it is equal to the action which it’s parent carried out.
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
                self._untried_actions = None
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

            def is_leaf_node(self):
                """
                Checks whether current node is leaf node
                """
                return len(self.children) == 0

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
                This function iteratively loops from child to parent and updates the amount of times visited.
                """
                self._number_of_visits += 1.
                self._results[result] += 1.
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
                choices_weights = []
                for c in self.children:
                    if c.no_visits() == 0:
                        if c.score() > 0:
                            choices_weights = math.inf
                        else:
                            choices_weights = -math.inf
                    else:
                        choices_weights = (c.score() / c.no_visits()) + c_param * np.sqrt((np.log(self.no_visits())) / c.no_visits())
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
                while not current_node.is_leaf_node():
                    current_node = current_node.best_child()

                if current_node.no_visits() != 0:
                    if not current_node.is_fully_expanded():
                        return current_node.expand()
                return current_node

        # here we check whether points can be achieved with the next move
        # if not and it looks like we are not going to end as the last player, change size by playing taboo move
        legal_moves_dict, greediest_move_tuple = compute_greediest_move(game_state)
        if len(get_all_empty_squares(game_state.board)) % 2 == 0 and greediest_move_tuple[3] == 0:
            taboo_moves = get_taboo_moves(game_state)
            while len(taboo_moves) != 0:
                taboo_move = random.choice(taboo_moves)
                if Move(taboo_move[0][0], taboo_move[0][1], taboo_move[1]) not in game_state.taboo_moves:
                    self.propose_move(Move(taboo_move[0][0], taboo_move[0][1], taboo_move[1]))
                    break

        # make a copy of the game_state so we do not ruin the original.
        game_state2 = copy.deepcopy(game_state)
        # Initialize root of the tree
        root = MonteCarloTreeSearchNode(game_state2, our_player_number=game_state.current_player() - 1)

        simulation_no = 1
        while True:
            v = root._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
            if simulation_no % 10 == 0:
                selected_node = root.best_child()
                self.propose_move(selected_node.parent_action)
            simulation_no += 1


