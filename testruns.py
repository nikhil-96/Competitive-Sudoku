import argparse
import importlib
import multiprocessing
import platform
import re
import time
from pathlib import Path
from competitive_sudoku.execute import solve_sudoku
from competitive_sudoku.sudoku import GameState, SudokuBoard, Move, TabooMove, load_sudoku_from_text
from competitive_sudoku.sudokuai import SudokuAI

import copy

from simulate_game import simulate_game


def main():
    solve_sudoku_path = 'bin\\solve_sudoku.exe'

    final_scores = {}
    # board_list = ['boards\\easy-2x2.txt', 'boards\\easy-3x3.txt', 'boards\\empty-2x2.txt', 'boards\\empty-2x3.txt',
    #               'boards\\empty-3x4.txt', 'boards\\empty-4x4.txt', 'boards\\hard-3x3.txt', 'boards\\random-2x3.txt',
    #               'boards\\random-3x3.txt', 'boards\\random-3x4.txt', 'boards\\random-4x4.txt']
    board_list = ['boards\\random-3x4.txt']

    calculation_time = [0.5, 1]


    for boardy in board_list:
        current_board = []
        board_text = Path(boardy).read_text()
        board = load_sudoku_from_text(board_text)
        evt_winrates = []
        evt_lossrates  = []
        # print("Checking board: ", boardy)
        for timey in calculation_time:
            time_scores = []
            for i in range(6):
                if i < 3:
                    module1 = importlib.import_module('team40_A2.sudokuai')
                    player1 = module1.SudokuAI()
                    module2 = importlib.import_module('greedy_player.sudokuai')
                    player2 = module2.SudokuAI()
                    player2.solve_sudoku_path = solve_sudoku_path

                else:
                    module1 = importlib.import_module('greedy_player.sudokuai')
                    player1 = module1.SudokuAI()
                    module2 = importlib.import_module('team40_A2.sudokuai')
                    player2 = module2.SudokuAI()
                    player1.solve_sudoku_path = solve_sudoku_path

                winner = simulate_game(board, player1, player2, solve_sudoku_path=solve_sudoku_path, calculation_time=timey)
                time_scores.append(winner)
                # print(time_scores)

            winrate = 0

            for j in range(6):
                # print(j)
                if j < 3 and time_scores[j]==1:
                    winrate = winrate +1
                elif j >= 3 and time_scores[j]==2:
                    winrate = winrate+1
            lossrate = (6 - winrate) / 6
            winrate = winrate/6

            evt_winrates.append(winrate)
            evt_lossrates.append(lossrate)


            print(evt_winrates)

        final_scores[boardy] = (evt_winrates, evt_lossrates)
        print(final_scores)

if __name__ == '__main__':
    main()