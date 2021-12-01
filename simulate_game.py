#!/usr/bin/env python3

#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

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


def check_oracle(solve_sudoku_path: str) -> None:
    board_text = '''2 2
       1   2   3   4
       3   4   .   2
       2   1   .   3
       .   .   .   1
    '''
    output = solve_sudoku(solve_sudoku_path, board_text)
    result = 'has a solution' in output
    if result:
        print('The sudoku_solve program works.')
    else:
        print('The sudoku_solve program gives unexpected results.')
        print(output)


def simulate_game(initial_board: SudokuBoard, player1: SudokuAI, player2: SudokuAI, solve_sudoku_path: str, calculation_time: float = 0.5) -> None:
    """
    Simulates a game between two instances of SudokuAI, starting in initial_board. The first move is played by player1.
    @param initial_board: The initial position of the game.
    @param player1: The AI of the first player.
    @param player2: The AI of the second player.
    @param solve_sudoku_path: The location of the oracle executable.
    @param calculation_time: The amount of time in seconds for computing the best move.
    """
    import copy
    N = initial_board.N

    game_state = GameState(initial_board, copy.deepcopy(initial_board), [], [], [0, 0])
    move_number = 0
    number_of_moves = initial_board.squares.count(SudokuBoard.empty)
    print('Initial state')
    print(game_state)

    with multiprocessing.Manager() as manager:
        # use a lock to protect assignments to best_move
        lock = multiprocessing.Lock()
        player1.lock = lock
        player2.lock = lock

        # use shared variables to store the best move
        player1.best_move = manager.list([0, 0, 0])
        player2.best_move = manager.list([0, 0, 0])

        while move_number < number_of_moves:
            player, player_number = (player1, 1) if len(game_state.moves) % 2 == 0 else (player2, 2)
            print(f'-----------------------------\nCalculate a move for player {player_number}')
            player.best_move[0] = 0
            player.best_move[1] = 0
            player.best_move[2] = 0
            try:
                process = multiprocessing.Process(target=player.compute_best_move, args=(game_state,))
                process.start()
                time.sleep(calculation_time)
                lock.acquire()
                process.terminate()
                lock.release()
            except Exception as err:
                print('Error: an exception occurred.\n', err)
            i, j, value = player.best_move
            best_move = Move(i, j, value)
            print(f'Best move: {best_move}')
            player_score = 0
            if best_move != Move(0, 0, 0):
                if TabooMove(i, j, value) in game_state.taboo_moves:
                    print(f'Error: {best_move} is a taboo move. Player {2-player_number} wins the game.')
                    return
                board_text = str(game_state.board)
                options = f'--move "{game_state.board.rc2f(i, j)} {value}"'
                output = solve_sudoku(solve_sudoku_path, board_text, options)
                if 'Invalid move' in output:
                    print(f'Error: {best_move} is not a valid move. Player {3-player_number} wins the game.')
                    return
                if 'Illegal move' in output:
                    print(f'Error: {best_move} is not a legal move. Player {3-player_number} wins the game.')
                    return
                if 'has no solution' in output:
                    print(f'The sudoku has no solution after the move {best_move}.')
                    player_score = 0
                    game_state.moves.append(TabooMove(i, j, value))
                    game_state.taboo_moves.append(TabooMove(i, j, value))
                if 'The score is' in output:
                    match = re.search(r'The score is ([-\d]+)', output)
                    if match:
                        player_score = int(match.group(1))
                        game_state.board.put(i, j, value)
                        game_state.moves.append(best_move)
                        move_number = move_number + 1
                    else:
                        raise RuntimeError(f'Unexpected output of sudoku solver: "{output}".')
            else:
                print(f'No move was supplied. Player {3-player_number} wins the game.')
                return
            game_state.scores[player_number-1] = game_state.scores[player_number-1] + player_score
            print(f'Reward: {player_score}')
            print(game_state)
        if game_state.scores[0] > game_state.scores[1]:
            print('Player 1 wins the game.')
        elif game_state.scores[0] == game_state.scores[1]:
            print('The game ends in a draw.')
        elif game_state.scores[0] < game_state.scores[1]:
            print('Player 2 wins the game.')


def main():
    solve_sudoku_path = 'bin\\solve_sudoku.exe' if platform.system() == 'Windows' else 'bin/solve_sudoku'

    cmdline_parser = argparse.ArgumentParser(description='Script for simulating a competitive sudoku game.')
    cmdline_parser.add_argument('--first', help="the module name of the first player's SudokuAI class (default: random_player)", default='random_player')
    cmdline_parser.add_argument('--second', help="the module name of the second player's SudokuAI class (default: random_player)", default='random_player')
    cmdline_parser.add_argument('--time', help="the time (in seconds) for computing a move (default: 0.5)", type=float, default=0.5)
    cmdline_parser.add_argument('--check', help="check if the solve_sudoku program works", action='store_true')
    cmdline_parser.add_argument('--board', metavar='FILE', type=str, help='a text file containing the start position')
    args = cmdline_parser.parse_args()

    if args.check:
        check_oracle(solve_sudoku_path)
        return

    board_text = '''2 2
       1   2   .   4
       .   4   .   2
       2   1   .   3
       .   .   .   1
    '''
    if args.board:
        board_text = Path(args.board).read_text()
    board = load_sudoku_from_text(board_text)

    module1 = importlib.import_module(args.first + '.sudokuai')
    module2 = importlib.import_module(args.second + '.sudokuai')
    player1 = module1.SudokuAI()
    player2 = module2.SudokuAI()
    if args.first in ('random_player', 'greedy_player'):
        player1.solve_sudoku_path = solve_sudoku_path
    if args.second in ('random_player', 'greedy_player'):
        player2.solve_sudoku_path = solve_sudoku_path

    simulate_game(board, player1, player2, solve_sudoku_path=solve_sudoku_path, calculation_time=args.time)


if __name__ == '__main__':
    main()
