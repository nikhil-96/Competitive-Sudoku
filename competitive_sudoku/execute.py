#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import os
from pathlib import Path
import tempfile


def execute_command(command: str) -> str:
    import subprocess
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as proc:
        output = proc.output
    return output.decode("utf-8").strip()


def solve_sudoku(solve_sudoku_path: str, board_text: str, options: str='') -> str:
    """
    Execute the solve_sudoku program.
    @param solve_sudoku_path: The location of the solve_sudoku executable.
    @param board_text: A string representation of a sudoku board.
    @param options: Additional command line options.
    @return: The output of solve_sudoku.
    """
    if not os.path.exists(solve_sudoku_path):
        raise RuntimeError(f'No oracle found at location "{solve_sudoku_path}"')
    filename = tempfile.NamedTemporaryFile(prefix='solve_sudoku_').name
    Path(filename).write_text(board_text)
    command = f'{solve_sudoku_path} {filename} {options}'
    return execute_command(command)
