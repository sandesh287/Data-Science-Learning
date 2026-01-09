# Deep Q-Networks (DQN)
# Deep Q-Networks (DQN) is an reinforcement learning algorithm that combines Q-Learning with deep neural networks. It uses a neural network to approximate the Q-values for each action in a given state, allowing it to handle environments with high-dimensional and continuous state spaces. DQN uses experience replay (storing past experiences and training on random batches) and a target network to stabilize training.




# libraries
import gymnasium as gym   # provides environment interface for reinforcement learning. Here we're going to use Cartpole version one
# These PyTorch libraries are used to build and train the Deep Q-Network (DQN)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random   # used here for sampling random actions and selecting batches from the replay buffer



# Define the DQN model
# This DQN class inherits from "nn.Module", making it a PyTorch neural network model
class DQN(nn.Module):
  # initialize the DQN model with three fully connected linear layers
  def __init__(self, input_size, output_size):
    super(DQN, self).__init__()   # calling super method
    self.linear1 = nn.Linear(input_size, 64)   # This is the first layer that takes input size, which is number of state features as the input and output 64 features
    self.linear2 = nn.Linear(64, 32)   # This is the second layer processes the 64 features and outputs 32 features
    self.linear3 = nn.Linear(32, output_size)   # This is the final layer that outputs "output_size" values, one for each possible action
  
  # creating forward method that defines the forward pass of the model
  def forward(self, x):
    x = torch.relu(self.linear1(x))   # applies ReLU activation to self.linear1 layers
    x = torch.relu(self.linear2(x))   # applies ReLU activation to self.linear2 layers
    return self.linear3(x)   # no activation on the third layer, as it outputs q-values directly



# Define hyperparameters
env_name = 'CartPole-v1'   # specifies gym environment
learning_rate = 0.001   # learning rate for Adam optimizer
gamma = 0.99   # discount factor for future rewards
buffer_size = 10000   # maximum number of experiences, to store in the replay buffer
batch_size = 32   # number of experiences sampled from the replay buffer per training step
epsilon = 0.1   # exploration rate for epsilon greedy action selection
target_update_frequency = 100   # number of steps between target network updates



# Initialize environment and DQN model
env = gym.make(env_name)   # env creates the gym environment specified by the environment name
input_size = env.observation_space.shape[0]   # sets the input size for the DQN model based on the state dimension
output_size = env.action_space.n   # sets the output size for the DQN model based on the number of possible actions


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # sets the computation device either GPU if it's available otherwise CPU

policy_net = DQN(input_size, output_size).to(device)   # This is the main DQN model for action selection
target_net = DQN(input_size, output_size).to(device)   # This is a secondary DQN model used to compute target Q-values and updated periodically for stability

target_net.load_state_dict(policy_net.state_dict())   # initialize a "target_net" with "policy_net" widgets. So, parameter that is passed is "policy_net.state_dictionary()"

target_net.eval()   # puts target_net in evaluation mode disabling gradient updates for stability


optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)   # creates Adam optimizer for updating "policy_nets" weights
criterion = nn.MSELoss()   # mean squared error loss for training which measures difference between predicted & target q-values



# Experience replay buffer
replay_buffer = []   # stores past experiences for experience replay. Experiences are tuples of state, action, reward, next_state, and done


# Training Function, which trains the agent over a specified number of episodes
def train(num_episodes):
  step_count = 0   # keeps track of the total steps for the target network updates
  
  for episode in range(num_episodes):   # loops through each episode for training
    state, _ = env.reset()   # resets the environment at the start of each episode
    state = np.array(state)   # ensures state is a numpy array, for later conversion to PyTorch tensors
    
    # state = np.array(env.reset())
    done = False   # initializes the done flag to keep track of episode
    total_reward = 0   # accumulates reward uh per episode
    
    while not done:   # inner loop where the agent interacts with the environment until the episode ends
      # Epsilon-greedy action selection
      # First with probability epsilon, choose a random action, otherwise use policy_net to choose the action with the highest predicted Q-value
      if random.random() < epsilon:
        action = env.action_space.sample()   # Exploration
      else:
        with torch.no_grad():
          action = policy_net(torch.tensor(state, dtype=torch.float, device=device)).argmax().item()
      
      
      # Take action and observe reward and next state
      next_state, reward, done, truncated, _ = env.step(action)   # takes the selected action receiving the next_state, reward and flags for completion
      
      done = done or truncated   # sets the done to true if either done or truncated is true. Ending the episode if the agent reaches the goal or the time limit.
      
      next_state = np.array(next_state)
      total_reward += reward   # updates cumulative reward for this particular episode
      
      
      # Store experience in replay buffer
      replay_buffer.append((state, action, reward, next_state, done))   # adds experience to replay buffer
      if len(replay_buffer) > buffer_size:
        replay_buffer.pop(0)   # remove the oldest experience
      
      
      # Update current state
      state = next_state   # updates state to the next state for the next step
      
      
      # Sample a batch from the replay buffer
      if len(replay_buffer) >= batch_size:   # ensures enough samples in the replay buffer before training
        batch = random.sample(replay_buffer, batch_size)   # samples mini batch from the buffer
        states, actions, rewards, next_states, dones = zip(*batch)   # unpacks the batch into separate components
        
        
        # Convert to tensors and move to device
        states = torch.tensor(states, dtype=torch.float, device=device)   # convert each batch component to a PyTorch tensor of the correct device
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float, device=device)
        dones = torch.tensor(dones, dtype=torch.float, device=device)
        
        
        # Compute the Q-values and target Q-values
        current_q_values = policy_net(states).gather(1, actions.unsqueeze(1))  # compute q-values for actions taken, using policy_net
        next_q_values = target_net(next_states).max(1)[0].detach()   # computes q-values of the next state using target_net
        target_q_values = rewards + gamma * next_q_values * (1 - dones)   # calculates the target q-values based on rewards and future q-values accounting for episode termination
        
        
        # Compute loss and update policy network
        loss = criterion(current_q_values, target_q_values.unsqueeze(1))   # calculates MSE between current and the target q-values
        optimizer.zero_grad()   # clears previous gradients
        loss.backward()   # computes the gradient
        optimizer.step()   # update model parameters
        
        
        # Update target network periodically
        step_count += 1
        if step_count % target_update_frequency == 0:
          target_net.load_state_dict(policy_net.state_dict())   # updates the target_net periodically with policy_net weights for stability
    
    
    print(f'Episode: {episode}, Total Reward: {total_reward}')



# Train the agent
train(num_episodes=1000)