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

    def tester(p1, p2, play1, board, board_text, time):

        num_runs = 5


        scores = []
        for i in range(num_runs):
            module1 = importlib.import_module(p1)
            player1 = module1.SudokuAI()
            module2 = importlib.import_module(p2)
            player2 = module2.SudokuAI()

            if play1:
                player2.solve_sudoku_path = solve_sudoku_path
            else:
                player1.solve_sudoku_path = solve_sudoku_path


            winner = simulate_game(board, player1, player2, solve_sudoku_path=solve_sudoku_path, calculation_time=timey)
            scores.append(winner)

        if play1:
            winrate = 0
            for i in range(num_runs):
                if scores[i] == 1:
                    winrate += 1
        else:
            winrate = 0
            for i in range(num_runs):
                if scores[i] == 2:
                    winrate += 1

        winscore = winrate / num_runs

        infolist = [time, play1, board_text, winscore]
        print(infolist)


    solve_sudoku_path = 'bin\\solve_sudoku.exe'

    # board_list = ['boards\\easy-2x2.txt', 'boards\\easy-3x3.txt', 'boards\\empty-2x2.txt', 'boards\\empty-2x3.txt',
    #               'boards\\empty-3x4.txt', 'boards\\empty-4x4.txt', 'boards\\hard-3x3.txt', 'boards\\random-2x3.txt',
    #               'boards\\random-3x3.txt', 'boards\\random-3x4.txt', 'boards\\random-4x4.txt']
    board_list = ['boards\\random-2x3.txt']

    calculation_time = [0.5, 1]

    for boardy in board_list:
        board_text = Path(boardy).read_text()
        board = load_sudoku_from_text(board_text)

        for timey in calculation_time:
            tester('team40_A2.sudokuai', 'greedy_player.sudokuai', True, board, boardy, timey)
            tester('greedy_player.sudokuai', 'team40_A2.sudokuai', False, board, boardy, timey)

if __name__ == '__main__':
    main()