#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import re
import numpy as np
from competitive_sudoku.execute import solve_sudoku
from competitive_sudoku.sudoku import GameState, Move
from datetime import datetime
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()
        self.solve_sudoku_path = None  # N.B. this path is set from outside

    # Uses solve_sudoku to compute a greedy move.
    def compute_best_move(self, game_state: GameState) -> None:
        '''
        Example code for load/save functionality
        '''
        #Create some random test data
        test_data = np.random.randint(low=1, high=10, size=10000000)

        #Save data
        self.save(test_data)

        #Load data
        saved_data = self.load()

        '''
        Random player functionality
        '''
        board = game_state.board
        board_text = str(board)
        options = '--random'
        taboo_moves = ' '.join(f'{move.i} {move.j} {move.value}' for move in game_state.taboo_moves)
        if taboo_moves:
            options += f' --taboo="{taboo_moves}"'
        output = solve_sudoku(self.solve_sudoku_path, board_text, options)
        m = re.search(r"Generated move \((\d+),(\d+)\)", output)
        if not m:
            raise RuntimeError('Could not generate a random move:\n' + output)
        k = int(m.group(1))
        value = int(m.group(2))
        i, j = board.f2rc(k)
        self.propose_move(Move(i, j, value))
