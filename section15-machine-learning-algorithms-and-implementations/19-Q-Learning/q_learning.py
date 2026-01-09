# Q-Learning
# Q-Learning is a model-free reinforcement learning algorithm used to find the optimal action-selection policy for a given problem. It learns by interacting with an environment, updating a Q-table (a matrix of state-action values), and maximizing the expected cumulative reward. Q-learning is effective in problems where the environment can be represented by discrete states and actions.




# libraries
import numpy as np   # used for numerical operations, including creating and manipulating q-table
import random   # imports python's random module for random choices within the code



# Define the environment (4x4 grid)
num_states = 16   # 4x4 grid; defines total number of states in 4x4 grid
num_actions = 4   # up, right, down, left; defines total number of possible actions. 0=up, 1=right, 2=down, 3=left
q_table = np.zeros((num_states, num_actions))   # initializes a q-table with zero values. It has 16 rows (one per state) and 4 columns (one per action), which will be updated with learned Q-values representing expected rewards for each state-action pair.



# Define the parameters
alpha = 0.1   # learning rate, which determines the extent to which new information overrides the old information
gamma = 0.9   # discount factor, which controls how much weight future rewards hold compared to immediate rewards
epsilon = 0.2   # exploration rate, which determines the likelihood of choosing a random action instead of the best known action
num_episodes = 1000   # number of episodes, which are iterations the agent will go through to learn



# Define a simple reward structure
rewards = np.zeros(num_states)   # initializes an array to store rewards for each state, initially setting all rewards to zero
rewards[15] = 1   # goal state with a reward, sets a reward of 1 for the goal state, which state equal to 15, while all other states remain at reward of 0



# Function to determine the next state based on the action
def get_next_state(state, action):
  if action == 0 and state >= 4:   # up action
    return state - 4
  elif action == 1 and (state + 1) % 4 != 0:   # right
    return state + 1
  elif action == 2 and state < 12:   # down moment
    return state + 4
  elif action == 3 and state % 4 != 0:   # left motion
    return state - 1
  else:
    return state   # if action goes out of bounds, remain in the same state

# This particular function "get_next_state" defines a function that calculates the next state based on the current state and the action, while ensuring the agent doesn't move out of the grid boundaries. The first action "zero", which is up, moves up if the agent is not in the top row. Action "one" right, moves right if the agent is not on the right edge. Action "two" down, moves down if the agent is not in the bottom row. Action "three" which is left, moves left if the agent is not on the left edge. And finally else, returns the same state if the action would go out of bounds.



# Q-Learning algorithm
for episode in range(num_episodes):   # this loops through each episode allowing the agent to learn iteratively
  state = random.randint(0, num_states - 1)   # start from random state; selects a random starting state for each episode
  while state != 15:   # loop until agent reaches the goal state, which is state of 15
    if random.uniform(0, 1) < epsilon:   # chooses action based on epsilon greedy policy, which allows exploration and exploitation
      action = random.randint(0, num_actions - 1)   # selects random action for exploration
    else:
      action = np.argmax(q_table[state])   # selects the best action by choosing the one with the highest q-value in the current state for exploration
    
    next_state = get_next_state(state, action)   # This calls "get_next_state" to get the new state based on the current state and chosen action
    reward = rewards[next_state]   # This retrieves the reward for moving to the next state
    old_value = q_table[state, action]   # This stores the current q-value of this state-action pair for future updates
    next_max = np.max(q_table[next_state])   # This gets the highest q-value of the next state to calculate expected future rewards
    
    
    # Q-learning update rule
    new_value = old_value + alpha + (reward + gamma * next_max - old_value)   # This updates q-value for current state-action pair using the Q-learning formula, which balances immediate and future rewards
    q_table[state, action] = new_value   # This stores the updated q-value in the q-table
    
    state = next_state   # This will move to the next state for next iteration, continuing until the goal state is reached



# Display the learned Q-table
print('Learned Q-Table:')
print(q_table)   # This displays the final Q-Table after training, showing learned q-values for each state-action pair. These values represent the expected reward for each action in each state learned over the episodes.