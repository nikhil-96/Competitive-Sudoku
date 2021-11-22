#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

from typing import List
from competitive_sudoku.sudoku import GameState, Move


class SudokuAI(object):
    """
    Sudoku AI that computes the best move in a given sudoku configuration.
    """

    def __init__(self):
        self.best_move: List[int] = [0, 0, 0]
        self.lock = None

    def compute_best_move(self, game_state: GameState) -> None:
        """
        This function should compute the best move in game_state.board. It should report the best move by making one
        or more calls to propose_move. This function is run by a game playing framework in a separate thread, that will
        be killed after a specific amount of time. The last reported move is the one that will be played.
        @param game_state: A Game state.
        """
        raise NotImplementedError

    def propose_move(self, move: Move) -> None:
        """
        Updates the best move that has been found so far.
        N.B. DO NOT CHANGE THIS FUNCTION!
        @param move: A move.
        """
        i, j, value = move.i, move.j, move.value
        if self.lock:
            self.lock.acquire()
        self.best_move[0] = i
        self.best_move[1] = j
        self.best_move[2] = value
        if self.lock:
            self.lock.release()
