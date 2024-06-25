class Sudoku:
    board = []
    evenNumbers = []
    def __init__(self):
        self.evenNumbers = []
        self.board = [[0 for _ in range(9)] for _ in range(9)]
        self.domains = {}


def initialization(matrix, array):
    sudoku = Sudoku()
    if len(matrix) != 9 or len(matrix[0]) != 9:
        return 'Matrix is not 9x9'
    sudoku.board = matrix
    sudoku.evenNumbers = array
    for i in range(9):
        for j in range(9):
            if sudoku.board[i][j] == 0:
                if (i, j) in sudoku.evenNumbers:
                    values = set(range(2, 10, 2))
                else:
                    values = set(range(1, 10))
                values = {value for value in values if valid_value(i, j, value, sudoku)}
                sudoku.domains[(i, j)] = values
            else:
                sudoku.domains[(i,j)] = {sudoku.board[i][j]}
    return sudoku

def valid_value(row, col, value, sudoku):
    if value in sudoku.board[row]:
        return False
    for i in range(9):
        if value == sudoku.board[i][col]:
            return False
    start_row, start_col = grid_start(row, col)
    for i in range(start_row, start_row+3):
        for j in range(start_col, start_col + 3):
            if sudoku.board[i][j] == value:
                return False
    return True

def grid_start(row, col):
    if row <= 2:
        start_row = 0
    elif row <= 5:
        start_row = 3
    else: start_row = 6
    if col <= 2:
        start_col = 0
    elif col <= 5:
        start_col = 3
    else:
        start_col = 6
    return start_row, start_col



def valid_move_with_forward_checking(row, col, value, sudoku):
    if (row, col) in sudoku.evenNumbers and value % 2 == 1:
        return False
    if valid_value(row, col, value, sudoku):
        old_domains = sudoku.domains.copy()
        update_domains(row, col, value, sudoku)
        if empty_domains(row, col, sudoku):
            sudoku.domains = old_domains
            return False
        return True
    return False


def empty_domains(row, col, sudoku):
    for i in range(9):
        if len(sudoku.domains[(row, i)]) == 0 or len(sudoku.domains[(i, col)]) == 0:
            return True
    start_row, start_col = grid_start(row, col)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if len(sudoku.domains[(i, j)]) == 0:
                return True
    return False

def update_domains(row, col, value, sudoku):
    for i in range(9):
        if sudoku.board[row][i] == 0:
            sudoku.domains[(row, i)].discard(value)
        if sudoku.board[i][col] == 0:
            sudoku.domains[(i, col)].discard(value)
    start_row, start_col = grid_start(row, col)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if sudoku.board[i][j] == 0:
                sudoku.domains[(i,j)].discard(value)


def is_final_solution(sudoku):
    for i in range(9):
        for j in range(9):
            if sudoku.board[i][j] == 0:
                return False
    return True

def empty_cell(sudoku):
    for i in range(9):
        for j in range(9):
            if sudoku.board[i][j] == 0:
                return i, j


def MRV(sudoku):
    variables = []
    for row in range(9):
        for col in range(9):
            if sudoku.board[row][col] == 0:
                variables.append((row, col))
    variables.sort(key=lambda var: len(sudoku.domains[var]))
    return variables[0]

def solve_sudoku(sudoku):
    if is_final_solution(sudoku):
        return True
    row, col = MRV(sudoku)
    for value in sudoku.domains[(row, col)]:
        if valid_value(row, col, value, sudoku):
            sudoku.board[row][col] = value
            if solve_sudoku(sudoku):
                return True
            sudoku.board[row][col] = 0

    return False


matrix = [
    [8, 4, 0, 0, 5, 0, 0, 0, 0],
    [3, 0, 0, 6, 0, 8, 0, 4, 0],
    [0, 0, 0, 4, 0, 9, 0, 0, 0],
    [0, 2, 3, 0, 0, 0, 9, 8, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 4],
    [0, 9, 8, 0, 0, 0, 1, 6, 0],
    [0, 0, 0, 5, 0, 3, 0, 0, 0],
    [0, 3, 0, 1, 0, 6, 0, 0, 7],
    [0, 0, 0, 0, 2, 0, 0, 1, 3]
]
even_numbers = [(0,6), (2,2), (2,8), (3,4), (4,3), (4,5), (5,4), (6,0), (6,6), (8,2)]

sudoku = initialization(matrix, even_numbers)
if solve_sudoku(sudoku):
    print("Soluție găsită:")
    for row in sudoku.board:
        print(row)
else:
    print("Nu există soluție pentru acest puzzle Sudoku.")