import numpy as np

rows, cols = 7, 10
wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
alpha = 0.5
gamma = 0.9
epsilon_initial = 1.0
epsilon_decay = 0.05


def take_action(state, action):
    row, col = state
    if action == 0:
        row = (max(0, row - 1 - wind[col])) % rows
    elif action == 1:
        row = (min(rows - 1, row + 1 - wind[col])) % rows
    elif action == 2:
        col = (col - 1) % cols
    elif action == 3:
        col = (col + 1) % cols
    return row, col


def q_learning(num_episodes, initial_state, final_state):
    Q = np.zeros((rows, cols, 4))
    epsilon = epsilon_initial
    for episode in range(num_episodes):
        current_state = initial_state
        while current_state != final_state:
            print("Current State:", current_state)
            if np.random.rand() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q[current_state[0], current_state[1]])

            next_state = take_action(current_state, action)
            reward = -1
            next_action = np.argmax(Q[next_state[0], next_state[1]])
            Q[current_state[0], current_state[1], action] += alpha * (
                        reward + gamma * Q[next_state[0], next_state[1], next_action] - Q[
                    current_state[0], current_state[1], action])
            current_state = next_state
        epsilon = max(0.1, epsilon - epsilon_decay)

    policy = np.argmax(Q, axis=2)

    print("Politica:")
    print(policy)

initial_state = (3, 0)
final_state = (3, 7)
num_episodes = 1000

q_learning(num_episodes, initial_state, final_state)