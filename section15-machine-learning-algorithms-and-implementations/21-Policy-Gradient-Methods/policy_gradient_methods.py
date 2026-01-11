# Policy Gradient Methods
# Policy Gradient Methods are a class of reinforcement learning algorithms that learn a policy directly by optimizing the parameters of a policy network. Instead of learning Q-values like Q-learning or DQN, policy gradient methods focus on finding the optimal action-selection strategy that maximizes cumulative rewards. A popular approach is the REINFORCE algorithm, where actions are sampled from a policy distribution, and the policy is updated using gradients based on rewards.




# libraries
import numpy as np
import tensorflow as tf   # for defining and training neural network
from keras import layers, Sequential
from keras.optimizers import Adam
import gymnasium as gym   # provides environment interface, here for CartPole-v1 environment



# Set up the environment
env = gym.make('CartPole-v1')
state_shape = env.observation_space.shape[0]   # gets number of state features (eg. position or velocity)
num_actions = int(env.action_space.n)   # defines the number of possible actions left or right in the cartpole environment



# Parameters
learning_rate = 0.01   # sets the step size for the optimizer to update the network weights
gamma = 0.99   # discount factor, determining how much future rewards are considered



# Policy network
# Function to define how to create the policy network
def build_policy_model():
  # sequential model where each layer feeds into the next
  model = Sequential([
    layers.Input(shape=(state_shape,)),
    layers.Dense(24, activation='relu'),   # first hidden layer with 24 neurons and ReLU activation
    layers.Dense(24, activation='relu'),   # second hidden layer with ReLU
    layers.Dense(num_actions, activation='softmax')   # output layer with number of actions output, for action probabilities and softmax activation to provide a probability distribution over actions
  ])
  
  model.compile(optimizer=Adam(learning_rate=learning_rate))   # compiles the model with Adam optimizer
  return model



# Call the function
policy_model = build_policy_model()   # call the model to create an instance of policy model that will be used to predict action probabilities



# Function to select an action based on policy network's output
def choose_action(state):
  state = np.array(state).reshape([1, state_shape])   # reshapes state into a 2D array with shape of (1, state_shape) for model input compatibility
  probabilities = policy_model.predict(state)   # uses the policy model to predict the probability distribution over actions
  return np.random.choice(num_actions, p=probabilities[0])   # chooses an action based on the predicted probabilities



# Function to calculate returns (discounted rewards)
def discount_rewards(rewards):
  discounted = np.zeros_like(rewards)   # initializes an array to store discounted rewards
  cumulative = 0
  for i in reversed(range(len(rewards))):   # iterates backward over the rewards
    cumulative = cumulative * gamma + rewards[i]   # updates the cumulative reward with discounting
    discounted[i] = cumulative   # stores the discounted reward
  return discounted - np.mean(discounted)   # normalizes the discounted reward by subtracting the mean to stabilize the training



# Training Function
# This function trains the model on data from a single episode
def train_on_episode(states, actions, rewards):
  discounted_rewards = discount_rewards(rewards)   # calculates the discounted rewards for the episode
  with tf.GradientTape() as tape:   # starts recording operations for automatic differentiation
    action_probs = policy_model(tf.convert_to_tensor(states, dtype=tf.float32), training=True)   # calculate action probabilities for each state in states
    action_indices = tf.stack([tf.range(len(actions)), actions], axis=1)   # prepare the indices to select the probability of the action taken
    selected_action_probs = tf.gather_nd(action_probs, action_indices)   # extract the action probabilities for the chosen actions
    loss = -tf.reduce_mean(tf.math.log(selected_action_probs) * discounted_rewards)   # calculates the policy gradient loss using the negative log probability of selected action weighted by discounted rewards
  
  gradients = tape.gradient(loss, policy_model.trainable_variables)   # computes gradients of the loss with respect to the model's parameters
  policy_model.optimizer.apply_gradients(zip(gradients, policy_model.trainable_variables))   # updates the model's parameters using the computed gradients



# Main training loop
num_episodes = 1000
for episode in range(num_episodes):
  state, _ = env.reset()   # resets the environment to start a new episode, and handles compatibility with the new gym version, where reset returns a tuple
  episode_states, episode_actions, episode_rewards = [], [], []
  while True:
    action = choose_action(state)   # selects an action using the policy
    next_state, reward, done, truncated, _ = env.step(action)   # takes the action in the environment and gets the resulting state
    done = done or truncated   # end the episode if done or truncated is true
    
    
    # stores the state chosen action and received rewards in their respective list for each step
    episode_states.append(state)
    episode_actions.append(action)
    episode_rewards.append(reward)
    
    state = next_state   # updates state to the next for the next step in the iteration
    
    if done:
      episode_states = np.vstack(episode_states)   # converts the episode_state into 2D numpy array
      train_on_episode(episode_states, np.array(episode_actions), np.array(episode_rewards))   # trains the model on the data
      
      print(f'Episode: {episode + 1}, Total Reward: {sum(episode_rewards)}')   # prints the log
      break







# This code trains a policy network using policy gradient reinforcement learning on the Cartpole-v1 environment and each episodes discounted rewards are used to update the policy network and actions are selected based on a probability distribution predicted by the network