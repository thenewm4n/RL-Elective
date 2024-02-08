import numpy as np
import copy

# GridWorld model
GRID_WIDTH = 4
GRID_HEIGHT = 3
TERMINAL_CELLS = [[3, 0], [3, 1]]
WALL = [[1, 1]]

# Hyperparameters
EXPLORATION_RATE = 0.1      # epsilon
LEARNING_RATE = 0.5         # alpha
DISCOUNT_FACTOR = 0.99      # gamma
NUM_EPISODES = 10000

actions = ("up", "right", "down", "left")

q_values = np.zeros((GRID_WIDTH, GRID_HEIGHT, len(actions)))     # first dimension is x; second dimension is y, third dimension is actions

for i in range(NUM_EPISODES):
    current_state = [0, 1]      # resets start position

    while current_state not in TERMINAL_CELLS:
        # If randomly sampled float between 0 and 1 is greater than epsilon, choose action randomly, else choose action greedily with respect to Q
        if np.random.rand() < EXPLORATION_RATE:
            current_action = np.random.choice(actions)
        else:
            current_action = actions[np.argmax(q_values[current_state[0], current_state[1], :])]
        
        # print("state: " + str(current_state[0]) + ", " + str(current_state[1]))
        # print("action: " + current_action)

        # Moves agent if not moving against wall
        next_state = copy.deepcopy(current_state)
        if current_action == "up":
            next_state[1] -= 1
        elif current_action == "right":
            next_state[0] += 1
        elif current_action == "down":
            next_state[1] += 1
        elif current_action == "left":
            next_state[0] -= 1

        # print("next state: " + str(next_state[0]) + ", " + str(next_state[1]))

        # If next state is invalid, next state remains the current state
        if next_state[0] < 0 or next_state[0] > GRID_WIDTH - 1 or next_state[1] < 0 or next_state[1] > GRID_HEIGHT - 1 or next_state in WALL:
            next_state = current_state
    
        # Observe reward
        immediate_reward = -1                               # for every timestep, -1 reward
        if next_state in TERMINAL_CELLS:
            if next_state == TERMINAL_CELLS[0]:             # TERMINAL_CELLS[0] has positive reward; TERMINAL_CELLS[1] has negative reward
                immediate_reward += 10
            else:
                immediate_reward -= 10
        
        # print("immediate reward: " + str(immediate_reward))
        # print()

        max_q_next_state = np.max(q_values[next_state[0], next_state[1], :])
        current_q = q_values[current_state[0], current_state[1], actions.index(current_action)]
        td_error = immediate_reward + DISCOUNT_FACTOR * max_q_next_state - current_q

        # Update Q of the previous state and the action just taken
        q_values[current_state[0], current_state[1], actions.index(current_action)] = current_q + LEARNING_RATE * td_error

        # S' -> S
        current_state = next_state

print()
for j in range(len(actions)):
    print(actions[j] + ":")
    for i in range(GRID_HEIGHT):
        print(q_values[:, i, j])
    print()
