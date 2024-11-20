import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import csv

# Constants
GRID_SIZE = 3
# (Other constants remain the same)

# Hyperparameters for DQN
learning_rate = 0.001
discount_factor = 0.99
exploration_rate = 1.0
exploration_decay = 0.999995
MIN_EXPLORATION_RATE = 0.01
BATCH_SIZE = 32
MEMORY_SIZE = 100000
TARGET_UPDATE_FREQ = 1000

# Game and training configurations
NUM_EPISODES = 500000
SHOW_EVERY = 100

# Define the neural network model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = x.float()
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)

# Initialize models and optimizer
state_size = 81 + 9 + 1
action_size = 81
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=MEMORY_SIZE)


# Functions to reset the board
def reset_board():
    board = np.zeros((GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE), dtype=int)
    main_board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    current_subgrid = None  # Start with no specific subgrid
    player = 1  # 1 for 'X', -1 for 'O'
    return board, main_board, current_subgrid, player

# Get available moves in the current subgrid
def get_available_moves(board, main_board, current_subgrid):
    moves = []
    if current_subgrid is not None and main_board[current_subgrid // GRID_SIZE, current_subgrid % GRID_SIZE] == 0:
        # Target subgrid is playable; return moves within it
        sg_row_start = (current_subgrid // GRID_SIZE) * GRID_SIZE
        sg_col_start = (current_subgrid % GRID_SIZE) * GRID_SIZE
        for i in range(sg_row_start, sg_row_start + GRID_SIZE):
            for j in range(sg_col_start, sg_col_start + GRID_SIZE):
                if board[i, j] == 0:
                    moves.append(i * GRID_SIZE * GRID_SIZE + j)
    else:
        # Target subgrid is not playable; can move in any playable subgrid
        for sg_row in range(GRID_SIZE):
            for sg_col in range(GRID_SIZE):
                if main_board[sg_row, sg_col] == 0:
                    sg_row_start = sg_row * GRID_SIZE
                    sg_col_start = sg_col * GRID_SIZE
                    for i in range(sg_row_start, sg_row_start + GRID_SIZE):
                        for j in range(sg_col_start, sg_col_start + GRID_SIZE):
                            if board[i, j] == 0:
                                moves.append(i * GRID_SIZE * GRID_SIZE + j)
    return moves

# Check for a winner in a subgrid or main board
def check_winner(grid):
    for i in range(GRID_SIZE):
        if np.all(grid[i, :] == grid[i, 0]) and grid[i, 0] != 0:
            return grid[i, 0]
        if np.all(grid[:, i] == grid[0, i]) and grid[0, i] != 0:
            return grid[0, i]
    if np.all(np.diag(grid) == grid[0, 0]) and grid[0, 0] != 0:
        return grid[0, 0]
    if np.all(np.diag(np.fliplr(grid)) == grid[0, GRID_SIZE - 1]) and grid[0, GRID_SIZE - 1] != 0:
        return grid[0, GRID_SIZE - 1]
    if np.all(grid != 0):
        return 'D'  # Draw
    return None

def is_game_over(main_board):
    return check_winner(main_board) is not None

# Update main board based on subgrid results
def update_main_board(board, main_board):
    for sg_row in range(GRID_SIZE):
        for sg_col in range(GRID_SIZE):
            if main_board[sg_row, sg_col] == 0:
                # Extract the subgrid
                sg_row_start = sg_row * GRID_SIZE
                sg_col_start = sg_col * GRID_SIZE
                subgrid = board[sg_row_start:sg_row_start + GRID_SIZE, sg_col_start:sg_col_start + GRID_SIZE]
                winner = check_winner(subgrid)
                if winner == 'D':
                    main_board[sg_row, sg_col] = 0.5  # Represent draw with 0.5
                elif winner:
                    main_board[sg_row, sg_col] = winner

# Choose a move using the epsilon-greedy strategy
def choose_action(state, available_moves):
    if np.random.rand() < exploration_rate:
        return random.choice(available_moves)
    else:
        with torch.no_grad():
            q_values = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).numpy().flatten()
        # Mask invalid actions
        masked_q_values = np.full_like(q_values, -np.inf)
        masked_q_values[available_moves] = q_values[available_moves]
        return np.argmax(masked_q_values)

# Prepare the state representation
def get_state(board, main_board, current_subgrid, player):
    # Flatten the board and main board and concatenate
    board_flat = board.flatten()
    main_board_flat = main_board.flatten()
    # Encode 'X' as 1, 'O' as -1, empty as 0
    state_board = board_flat * player
    state_main_board = main_board_flat * player
    # Include current subgrid in the state
    if current_subgrid is not None:
        current_subgrid_state = np.array([current_subgrid])
    else:
        current_subgrid_state = np.array([-1])  # Use -1 to represent no specific subgrid
    state = np.concatenate([state_board, state_main_board, current_subgrid_state])
    return state

# Replay memory sampling
def sample_memory():
    batch = random.sample(memory, min(len(memory), BATCH_SIZE))
    states, actions, rewards, next_states, dones = zip(*batch)
    return np.array(states), actions, rewards, np.array(next_states), dones

# Initialize metrics
mse_list = []
cumulative_rewards_list = []
win_loss_ratio_list = []
exploration_rates_list = []

# Training loop
wins = {1: 0, -1: 0, 'D': 0}
cumulative_rewards = 0
steps_done = 0

for episode in range(1, NUM_EPISODES + 1):
    board, main_board, current_subgrid, player = reset_board()
    done = False
    current_player = player
    state = get_state(board, main_board, current_subgrid, current_player)
    episode_rewards = 0  # Track rewards per episode
    
    while not done:
        available_moves = get_available_moves(board, main_board, current_subgrid)
        if not available_moves:
            # No moves available, it's a draw
            winner = 'D'
            wins['D'] += 1
            break
        
        action = choose_action(state, available_moves)
        row, col = divmod(action, GRID_SIZE * GRID_SIZE)
        
        # Apply the action
        board[row, col] = current_player
        
        # Update current subgrid based on the move
        cell_row_in_subgrid = row % GRID_SIZE
        cell_col_in_subgrid = col % GRID_SIZE
        new_subgrid = cell_row_in_subgrid * GRID_SIZE + cell_col_in_subgrid
        
        # Update the main board
        update_main_board(board, main_board)
        
        # Check for game over
        winner = check_winner(main_board)
        if winner == current_player:
            reward = 1
            wins[current_player] += 1
            done = True
        elif winner == 'D':
            reward = 0
            wins['D'] += 1
            done = True
        else:
            reward = 0
            done = False
        
        episode_rewards += reward
        
        # Get the next state
        if main_board.flatten()[new_subgrid] == 0:
            next_current_subgrid = new_subgrid
        else:
            next_current_subgrid = None  # Any subgrid
        
        next_state = get_state(board, main_board, next_current_subgrid, -current_player)
        
        # Store the experience in replay memory
        memory.append((state, action, reward, next_state, done))
        
        state = next_state
        current_subgrid = next_current_subgrid
        current_player = -current_player  # Switch player
        
        # Optimize the model
        if len(memory) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = sample_memory()
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)
            
            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values = target_net(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * discount_factor * next_q_values
            loss = nn.functional.mse_loss(q_values, target_q_values)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            mse_list.append(loss.item())  # Track MSE
        
        steps_done += 1
        
        # Update the target network
        if steps_done % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    cumulative_rewards += episode_rewards
    cumulative_rewards_list.append(cumulative_rewards)
    
    if wins[-1] + wins[1] > 0:
        win_loss_ratio = wins[1] / (wins[-1] + wins[1])
    else:
        win_loss_ratio = 0
    win_loss_ratio_list.append(win_loss_ratio)
    
    exploration_rates_list.append(exploration_rate)
    
    # Decay exploration rate
    exploration_rate = max(MIN_EXPLORATION_RATE, exploration_rate * exploration_decay)
    
    # Display progress
    if episode % SHOW_EVERY == 0:
        print(f"Episode {episode}, Wins: X = {wins[1]}, O = {wins[-1]}, Draws = {wins['D']}, Exploration Rate: {exploration_rate:.4f}")

# Plot and save metrics
def plot_and_save(metric, values, filename, ylabel):
    plt.figure()
    plt.plot(values)
    plt.title(f'{metric} Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

plot_and_save('MSE', mse_list, 'mse_plot.png', 'Mean Squared Error')
plot_and_save('Cumulative Rewards', cumulative_rewards_list, 'cumulative_rewards_plot.png', 'Cumulative Rewards')
plot_and_save('Win/Loss Ratio', win_loss_ratio_list, 'win_loss_ratio_plot.png', 'Win/Loss Ratio')
plot_and_save('Exploration vs Exploitation', exploration_rates_list, 'exploration_rate_plot.png', 'Exploration Rate')

print("Metrics plots saved to files.")

# End of training
print("Training complete!")
print(f"Final win counts: X = {wins[1]}, O = {wins[-1]}, Draws = {wins['D']}")

# Save the trained policy network model
torch.save(policy_net.state_dict(), 'trained_policy_net.pth')
print("Trained model saved as 'trained_policy_net.pth'")

# Save the training results to a CSV file
with open('training_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Player', 'Wins'])  # Header
    writer.writerow(['X', wins[1]])
    writer.writerow(['O', wins[-1]])
    writer.writerow(['Draws', wins['D']])

print("Training results saved as 'training_results.csv'")