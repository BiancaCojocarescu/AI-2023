import heapq
import math
import timeit


class State:
    def __init__(self, matrix=None, cant_be_moved=None, distance_method=None):
        self.matrix = matrix if matrix else [[0 for _ in range(3)] for _ in range(3)]
        self.cant_be_moved = cant_be_moved if cant_be_moved else []
        self.distance_method = distance_method

    def __str__(self):
        matrix_str = "\n".join([" ".join(map(str, row)) for row in self.matrix])
        return f"Matrix: \n{matrix_str}\n"

    def __lt__(self, other):
        if self.distance_method == "manhattan":
            return manhattan_distance(self) < manhattan_distance(other)
        elif self.distance_method == "euclidean":
            return euclidean_distance(self) < euclidean_distance(other)
        elif self.distance_method == "hamming":
            return hamming_distance(self) < hamming_distance(other)
        else:
            raise ValueError("Invalid distance method")


def initialization(problem_instance, comparision_type):
    row = []
    instances = problem_instance.split()
    matrix = []

    for instance in instances:
        row.append(int(instance))
        if len(row) == 3:
            matrix.append(row)
            row = []

    return State(matrix=matrix, distance_method=comparision_type)


def was_moved(state, row, col):
    element = state.matrix[row][col]
    return element in state.cant_be_moved


def get_neighbors(state, row, col):
    rows = len(state.matrix)
    cols = len(state.matrix[0])
    neighbours = [-1, -1, -1, -1]

    if col > 0:
        neighbours[0] = state.matrix[row][col - 1]
    if row > 0:
        neighbours[1] = state.matrix[row - 1][col]
    if col < cols - 1:
        neighbours[2] = state.matrix[row][col + 1]
    if row < rows - 1:
        neighbours[3] = state.matrix[row + 1][col]

    return neighbours


def is_valid_transition(state, row, col):
    if was_moved(state, row, col):
        return False
    elif state.matrix[row][col] == 0:
        return False
    elif 0 in get_neighbors(state, row, col):
        return True
    else:
        return False


def move_to_left(state, row, col):
    aux = state.matrix[row][col]
    state.matrix[row][col] = state.matrix[row][col - 1]
    state.matrix[row][col - 1] = aux


def move_to_right(state, row, col):
    aux = state.matrix[row][col]
    state.matrix[row][col] = state.matrix[row][col + 1]
    state.matrix[row][col + 1] = aux


def move_up(state, row, col):
    aux = state.matrix[row][col]
    state.matrix[row][col] = state.matrix[row - 1][col]
    state.matrix[row - 1][col] = aux


def move_down(state, row, col):
    aux = state.matrix[row][col]
    state.matrix[row][col] = state.matrix[row + 1][col]
    state.matrix[row + 1][col] = aux


def transition(state, row, col, distance_method):
    if is_valid_transition(state, row, col):
        for val in state.cant_be_moved:
            if val in get_neighbors(state, row, col):
                state.cant_be_moved.remove(val)
        state.cant_be_moved.append(state.matrix[row][col])
        neighbours = get_neighbors(state, row, col)
        for i in range(4):
            if neighbours[i] == 0:
                break
        if i == 0:
            move_to_left(state, row, col)
        elif i == 1:
            move_up(state, row, col)
        elif i == 2:
            move_to_right(state, row, col)
        else:
            move_down(state, row, col)
        return state


def is_final_state(state):
    rows = len(state.matrix)
    cols = len(state.matrix[0])
    elements = [state.matrix[i][j] for i in range(rows) for j in range(cols) if state.matrix[i][j] != 0]
    elements_sorted = sorted(elements)
    return elements == elements_sorted


def dls(state, maxDepth):
    stack = [(state, 0)]
    moves = 0
    while stack:
        current_state, depth = stack.pop()
        # print(current_state)
        if depth <= maxDepth:
            if is_final_state(current_state):
                return current_state, moves
            for i in range(3):
                for j in range(3):
                    if is_valid_transition(current_state, i, j):
                        next_state = transition(State(matrix=[row[:] for row in current_state.matrix]), i, j, None)
                        moves += 1
                        stack.insert(0, (next_state, depth + 1))
    return None, None


def iddfs(problem_instance, maxDepth):
    initial_state = initialization(problem_instance, None)
    for depth in range(maxDepth + 1):
        result, moves = dls(initial_state, depth)
        if result:
            return result, moves
    return None, None


def manhattan_distance(state):
    distance = 0
    for i in range(len(state.matrix)):
        for j in range(len(state.matrix[0])):
            if state.matrix[i][j] != 0:
                target_row1 = (state.matrix[i][j] - 1) // 3
                target_col1 = (state.matrix[i][j] - 1) % 3
                target_row2 = state.matrix[i][j] // 3
                target_col2 = state.matrix[i][j] % 3
                d1 = abs(i - target_row1) + abs(j - target_col1)
                d2 = abs(i - target_row2) + abs(j - target_col2)
                distance += min(d1, d2)
    return distance


def euclidean_distance(state):
    distance = 0
    for i in range(len(state.matrix)):
        for j in range(len(state.matrix[0])):
            if state.matrix[i][j] != 0:
                target_row1 = (state.matrix[i][j] - 1) // 3
                target_col1 = (state.matrix[i][j] - 1) % 3
                target_row2 = state.matrix[i][j] // 3
                target_col2 = state.matrix[i][j] % 3
                d1 = math.sqrt((i - target_row1) ** 2 + (j - target_col1) ** 2)
                d2 = math.sqrt((i - target_row2) ** 2 + (j - target_col2) ** 2)
                distance += min(d1, d2)
    return distance


def hamming_distance(state):
    distance = 0
    for i in range(len(state.matrix)):
        for j in range(len(state.matrix[0])):
            if state.matrix[i][j] != 0:
                target_row1 = (state.matrix[i][j] - 1) // 3
                target_col1 = (state.matrix[i][j] - 1) % 3
                target_row2 = state.matrix[i][j] // 3
                target_col2 = state.matrix[i][j] % 3
                if i != target_row1 and i != target_row2:
                    distance += 1
                elif j != target_col1 and j != target_col2:
                    distance += 1
    return distance


def calculate_distance(state, distance_method):
    if distance_method == "manhattan":
        return manhattan_distance(state)
    elif distance_method == "euclidean":
        return euclidean_distance(state)
    elif distance_method == "hamming":
        return hamming_distance(state)
    else:
        raise ValueError("Invalid distance method")


def greedy(init_state, distance_method):
    priority_queue = []
    visited = set()
    moves = 0
    heapq.heappush(priority_queue, (calculate_distance(init_state, distance_method), init_state))
    visited.add(init_state)
    while priority_queue:
        _, state = heapq.heappop(priority_queue)
        if is_final_state(state):
            return state, moves
        for i in range(3):
            for j in range(3):
                if is_valid_transition(state, i, j):
                    neighbor = transition(
                        State(matrix=[row[:] for row in state.matrix], distance_method=distance_method), i, j,
                        distance_method)
                    moves += 1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        heapq.heappush(priority_queue, (calculate_distance(neighbor, distance_method), neighbor))
    return None, None


def search_strategies(problem_instance, max_depth):
    initial_state = initialization(problem_instance, None)

    def run_greedy(distance_method):
        def inner_function():
            nonlocal distance_method
            result_state = greedy(initial_state, distance_method)
            if result_state:
                print(f"Greedy ({distance_method}) found a solution \n{result_state[0]} in {result_state[1]} moves.")

        execution_time = timeit.timeit(inner_function, number=1)
        print(f"Time taken by Greedy ({distance_method}): {execution_time} seconds")

    def run_iddfs():
        def inner_function():
            result_state = iddfs(problem_instance, max_depth)
            if result_state:
                print(
                    f"IDDFS found a solution \n{result_state[0]} in {result_state[1]} moves.")

        execution_time = timeit.timeit(inner_function, number=1)
        print(f"Time taken by IDDFS: {execution_time} seconds")

    run_iddfs()
    for distance_method in ["manhattan", "euclidean", "hamming"]:
        run_greedy(distance_method)


problem_instance = "2 5 3 1 0 6 4 7 8"
max_depth = 50

search_strategies(problem_instance, max_depth)
