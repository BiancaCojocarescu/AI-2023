def init_game():
    return [[2, 7, 6], [9, 5, 1], [4, 3, 8]]

def print_board(board):
    for row in board:
        print(row)

def available_move(player, board):
    move = int(input(f"{player}, introduceți un numar intre 1 și 9: "))
    if move < 1 or move > 9:
        return None
    for row in board:
         if move in row:
             return move
    return None

def make_move(player, move, board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == move:
                board[i][j] = player

def check_win(player, board):
    for row in board:
        if row.count(player) == 3:
            return True
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] == player:
            return True
    if board[0][0] == board[1][1] == board[2][2] == player:
        return True
    if board[0][2] == board[1][1] == board[2][0] == player:
        return True
    return False

def heuristic(board):
    score = 0
    #daca am un rand, coloana sau diagonala completa cu B sau cu A
    for row in board:
        if row.count('B') == 3:
            score += 100
        elif row.count('A') == 3:
            score -= 100
    for col in range(3):
        col_values = [board[row][col] for row in range(3)]
        if col_values.count('B') == 3:
            score += 100
        elif col_values.count('A') == 3:
            score -= 100
    if board[0][0] == board[1][1] == board[2][2] == 'B':
        score += 100
    elif board[0][0] == board[1][1] == board[2][2] == 'A':
        score -= 100
    if board[0][2] == board[1][1] == board[2][0] == 'B':
        score += 100
    elif board[0][2] == board[1][1] == board[2][0] == 'A':
        score -= 100
    #daca am 2 de B sau 2 de A pe rand, coloana sau diagonala
    for row in board:
        if row.count('B') == 2 and row.count(0) == 1:
            score += 10
        elif row.count('A') == 2 and row.count('B') == 0:
            score -= 10
    for col in range(3):
        col_values = [board[row][col] for row in range(3)]
        if col_values.count('B') == 2 and col_values.count('A') == 0:
            score += 10
        elif col_values.count('A') == 2 and col_values.count('B') == 0:
            score -= 10
    diag1_values = [board[row][row] for row in range(3) ]
    if diag1_values.count('B') == 2 and diag1_values.count('A') == 0:
        score += 10
    elif  diag1_values.count('A') == 2 and diag1_values.count('B') == 0:
        score -= 10
    diag2_values = []
    diag2_values.append(board[0][2])
    diag2_values.append(board[2][0])
    diag2_values.append(board[1][1])
    if diag2_values.count('B') == 2 and diag2_values.count('A') == 0:
        score += 10
    elif diag2_values.count('A') == 2 and diag2_values.count('B') == 0:
        score -= 10

    return score

def minimax(depth, board, maximizing_player):
    if check_win("B", board):
        return 100
    elif check_win("A", board):
        return -100
    elif depth == 0:
        return heuristic(board)

    if maximizing_player:
        max_eval = float("-inf")
        for move in range(1, 10):
            if move in [cell for row in board for cell in row]:
                i, j = [(i, j) for i in range(3) for j in range(3) if board[i][j] == move][0]
                board[i][j] = "B"
                eval = minimax(depth - 1, board, False)
                board[i][j] = move
                max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float("inf")
        for move in range(1, 10):
            if move in [cell for row in board for cell in row]:
                i, j = [(i, j) for i in range(3) for j in range(3) if board[i][j] == move][0]
                board[i][j] = "A"
                eval = minimax(depth - 1, board, True)
                board[i][j] = move
                min_eval = min(min_eval, eval)
        return min_eval


def best_move_with_min_max(board, depth):
    max_score = float("-inf")
    best_move = None
    for move in range(1, 10):
        if move in [cell for row in board for cell in row]:
            i, j = [(i, j) for i in range(3) for j in range(3) if board[i][j] == move][0]
            board[i][j] = "B"
            eval = minimax(depth - 1, board, False)
            board[i][j] = move
            if eval > max_score:
                max_score = eval
                best_move = move
    return best_move

def play_game_with_min_max(depth):
    board = init_game()
    players = ["A", "B"]
    current_player = 0
    while True:
        print_board(board)
        if players[current_player] == "A":
            move = available_move(players[current_player], board)
        else:
            move = best_move_with_min_max(board, depth)
            print(f"Calculatorul a ales: {move}")
        if move is None:
            continue
        else:
            make_move(players[current_player], move, board)
            if check_win(players[current_player], board):
                print(f"{players[current_player]} a castigat!")
                return
            if all(isinstance(x, str) for row in board for x in row):
                print("Remiza!")
                return
            current_player = (current_player + 1) % 2

play_game_with_min_max(2)
