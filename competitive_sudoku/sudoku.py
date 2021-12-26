#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

from typing import List, Tuple, Union



class Move(object):
    """A Move is a tuple (i, j, value) that represents the action board.put(i, j, value) for a given
    sudoku configuration board."""

    def __init__(self, i: int, j: int, value: int):
        """
        Constructs a move.
        @param i: A row value in the range [0, ..., N)
        @param j: A column value in the range [0, ..., N)
        @param value: A value in the range [1, ..., N]
        """
        self.i = i
        self.j = j
        self.value = value

    def __str__(self):
        return f'({self.i},{self.j}) -> {self.value}'

    def __eq__(self, other):
        return (self.i, self.j, self.value) == (other.i, other.j, other.value)


class TabooMove(Move):
    """A TabooMove is a Move that was flagged as illegal by the sudoku oracle. In other words, the execution of such a
    move would cause the sudoku to become unsolvable.
    """

    """
    Constructs a taboo move.
    @param i: A row value in the range [0, ..., N)
    @param j: A column value in the range [0, ..., N)
    @param value: A value in the range [1, ..., N]
    """
    def __init__(self, i: int, j: int, value: int):
        super().__init__(i, j, value)


class SudokuBoard(object):
    """
    A simple board class for Sudoku. It supports arbitrary rectangular regions.
    """

    empty = 0  # Empty squares contain the value SudokuBoard.empty

    def __init__(self, m: int = 3, n: int = 3):
        """
        Constructs an empty Sudoku with regions of size m x n.
        @param m: The number of rows in a region.
        @param n: The number of columns in a region.
        """
        N = m * n
        self.m = m
        self.n = n
        self.N = N     # N = m * n, numbers are in the range [1, ..., N]
        self.squares = [SudokuBoard.empty] * (N * N)  # The N*N squares of the board

    def rc2f(self, i: int, j: int):
        """
        Converts row/column coordinates to the corresponding index in the board array.
        @param i: A row value in the range [0, ..., N)
        @param j: A column value in the range [0, ..., N)
        @return: The corresponding index k in the board array
        """
        N = self.N
        return N * i + j

    def f2rc(self, k: int) -> Tuple[int, int]:
        """
        Converts an index in the board array to the corresponding row/column coordinates.
        @param k: A value in the range [0, ..., N * N)
        @return: The corresponding row/column coordinates
        """
        N = self.N
        i = k // N
        j = k % N
        return i, j

    def put(self, i: int, j: int, value: int) -> None:
        """
        Puts the given value on the square with coordinates (i, j).
        @param i: A row value in the range [0, ..., N)
        @param j: A column value in the range [0, ..., N)
        @param value: A value in the range [1, ..., N]
        """
        k = self.rc2f(i, j)
        self.squares[k] = value

    def get(self, i: int, j: int):
        """
        Gets the value of the square with coordinates (i, j).
        @param i: A row value in the range [0, ..., N)
        @param j: A column value in the range [0, ..., N)
        @return: The value of the square.
        """
        k = self.rc2f(i, j)
        return self.squares[k]

    def region_width(self):
        """
        Gets the number of columns in a region.
        @return: The number of columns in a region.
        """
        return self.n

    def region_height(self):
        """
        Gets the number of rows in a region.
        @return: The number of rows in a region.
        """
        return self.m

    def board_width(self):
        """
        Gets the number of columns of the board.
        @return: The number of columns of the board.
        """
        return self.N

    def board_height(self):
        """
        Gets the number of rows of the board.
        @return: The number of rows of the board.
        """
        return self.N

    def __str__(self) -> str:
        """
        Prints the board in a simple textual format. The first line contains the values m and n. Then the contents of
        the rows are printed as space separated lists, where a dot '.' is used to represent an empty square.
        @return: The generated string.
        """
        import io

        m = self.m
        n = self.n
        N = self.N
        out = io.StringIO()

        def print_square(i, j):
            value = self.get(i, j)
            s = '   .' if value == 0 else f'{value:>4}'
            out.write(s)

        out.write(f'{m} {n}\n')
        for i in range(N):
            for j in range(N):
                print_square(i, j)
            out.write('\n')
        return out.getvalue()


# written by Gennaro Gala
def print_board(board: SudokuBoard) -> str:
    import io

    m = board.m
    n = board.n
    N = board.N
    out = io.StringIO()

    def print_square(i, j):
        value = board.get(i, j)
        s = ' -' if value == 0 else f'{value:2}'
        return s

    for i in range(N):

        # open the grid
        if i == 0:
            out.write('  ')
            for j in range(N):
                out.write(f'   {j}  ')
            out.write('\n')
            for j in range(N):
                if j % n != 0:
                    out.write('╤═════')
                elif j != 0:
                    out.write('╦═════')
                else:
                    out.write('   ╔═════')
            out.write('╗\n')

        # separate regions horizontally
        if i % m == 0 and i != 0:
            for j in range(N):
                if j % n != 0:
                    out.write('╪═════')
                elif j != 0:
                    out.write('╬═════')
                else:
                    out.write('   ╠═════')
            out.write('║\n')

        # plot values
        out.write(f'{i:2} ')
        for j in range(N):
            symbol = print_square(i, j)
            if j % n != 0:
                out.write(f'│ {symbol}  ')
            else:
                out.write(f'║ {symbol}  ')
            if len(symbol) < 2:
                out.write(' ')
        out.write('║\n')

        # close the grid
        if i == N - 1:
            for j in range(N):
                if j % n != 0:
                    out.write('╧═════')
                elif j != 0:
                    out.write('╩═════')
                else:
                    out.write('   ╚═════')
            out.write('╝\n')

    return out.getvalue()


def load_sudoku_from_text(text: str) -> SudokuBoard:
    """
    Loads a sudoku board from a string, in the same format as used by the SudokuBoard.__str__ function.
    @param text: A string representation of a sudoku board.
    @return: The generated Sudoku board.
    """
    words = text.split()
    if len(words) < 2:
        raise RuntimeError('The string does not contain a sudoku board')
    m = int(words[0])
    n = int(words[1])
    N = m * n
    if len(words) != N*N + 2:
        raise RuntimeError('The number of squares in the sudoku is incorrect.')
    result = SudokuBoard(m, n)
    N = result.N
    for k in range(N * N):
        s = words[k + 2]
        if s != '.':
            value = int(s)
            result.squares[k] = value
    return result


def load_sudoku(filename: str) -> SudokuBoard:
    """
    Loads a sudoku board from a file, in the same format as used by the SudokuBoard.__str__ function.
    @param filename: A file name.
    @return: The generated Sudoku board.
    """
    from pathlib import Path
    text = Path(filename).read_text()
    return load_sudoku_from_text(text)


def save_sudoku(filename, board: SudokuBoard) -> None:
    """
    Saves a sudoku board to a file, in the same format as used by the SudokuBoard.__str__ function.
    @param filename: A file name.
    @param board: A sudoku board.
    @return: The generated Sudoku board.
    """
    from pathlib import Path
    Path(filename).write_text(str(board))


class GameState(object):
    def __init__(self,
                 initial_board: SudokuBoard,
                 board: SudokuBoard,
                 taboo_moves: List[TabooMove],
                 moves: List[Union[Move, TabooMove]],
                 scores: List[int]):
        """
        @param initial_board: A sudoku board. It contains the start position of a game.
        @param board: A sudoku board. It contains the current position of a game.
        @param taboo_moves: A list of taboo moves. Moves in this list cannot be played.
        @param moves: The history of a sudoku game, starting in initial_board. The
        history includes taboo moves.
        @param scores: The current scores of the first and the second player.
        """
        self.initial_board = initial_board
        self.board = board
        self.taboo_moves = taboo_moves
        self.moves = moves
        self.scores = scores

    def current_player(self):
        """Gives the index of the current player (1 or 2). The convention is that player 1
        does the first move of the game.
        @return The index of the current player.
        """
        return 1 if len(self.moves) % 2 == 0 else 2

    def __str__(self):
        import io
        out = io.StringIO()
        out.write(print_board(self.board))
        out.write(f'Score: {self.scores[0]} - {self.scores[1]}')
        return out.getvalue()
