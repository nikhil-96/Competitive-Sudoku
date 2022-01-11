#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

from typing import List
from competitive_sudoku.sudoku import GameState, Move
import os
import pickle
import math
from datetime import datetime


class SudokuAI(object):
    """
    Sudoku AI that computes the best move in a given sudoku configuration.
    """

    def __init__(self):
        self.best_move: List[int] = [0, 0, 0]
        self.lock = None
        self.player_number = -1

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

    def save(self, object):
        if self.lock:
            self.lock.acquire()
        save_path = os.path.join(os.getcwd(), '{}.pkl'.format(self.player_number))
        start_time = datetime.now()
        with open(save_path, 'wb') as handle:
            pickle.dump(object, handle)
            handle.close()
        end_time = datetime.now()
        duration =  end_time - start_time
        # print('Saving data took {} seconds and {} milliseconds'.format(math.floor(duration.total_seconds()), round(duration.microseconds/1000)))
        if self.lock:
            self.lock.release()


    def load(self):
        if self.lock:
            self.lock.acquire()
        load_path = os.path.join(os.getcwd(), '{}.pkl'.format(self.player_number))
        start_time = datetime.now()
        if not os.path.isfile(load_path):
            if self.lock:
                self.lock.release()
            return None
        with open(load_path, 'rb') as handle:
            contents = pickle.load(handle)
            handle.close()
            end_time = datetime.now()
            duration =  end_time - start_time
            # print('Loading data took {} seconds and {} milliseconds'.format(math.floor(duration.total_seconds()), round(duration.microseconds/1000)))
        if self.lock:
            self.lock.release()
        return contents
