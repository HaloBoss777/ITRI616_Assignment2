#import gymasium as gym
import gymnasium as gym

#import numpy for Q table
import numpy as np

#Custom Environments
#S = Start, F = floor, H = hole, G = Goal
from maps import maps

#Import graphing library
import matplotlib.pyplot as plt

#Import for Q table to dump into a binary file
import pickle


#Make a function to find the start of a map
def find_position_of_s(map_array): # For 4x4: S is index 12 from 0
    for i, row in enumerate(map_array):
        for j, char in enumerate(row):
            if char == 'S':
                return i * len(row) + j
    return None  # Return None if 'S' is not found


#Create function for eayer calling
def FrozenLake(Custom_Map, Episodes):

  # make a environment object, of forzen lake physics, custom maps, enable slippery flag, turn on rendering
  env = gym.make('FrozenLake-v1', desc=Custom_Map, is_slippery=True, render_mode= None)

  """
  is_slippery=True creates randomness in the environment, 
  which makes it harder for the agent to navigate by needing more complex search algorithms 
  or reinforcement learning techniques.
  """

  #Initalize Q - table (64 states and 4 action) (Full of zeros)
  q = np.zeros((env.observation_space.n, env.action_space.n)) #zeros expects a tuple need to add parentheses around dimensions to make it a tuple 

  #Used to update the Q-Learning Formula
  learning_rate_a = 0.8 #Alpha or Learing rate (α is the learning rate (0 < α <= 1))
  discount_factor_g = 0.8 # gamma or discount factor (γ is the discount factor (0 <= γ < 1), which determines the importance of future rewards)

  #Implement Epsilon-Greedy algorithm, allow the follow of the path dictated by the Q table
  epsilon = 1 # When 1 means we are picking 100% random actions
  epsilon_decay_rate = 0.0001 # Decrease the randomness over thime (1/0.0001 = 10 000 minumim episodes to get epsilon 0)

  #Declaring a random number generator
  rng = np.random.default_rng()  #Generate a number between 0.0 and 1.0

  #Reward per episode:
  rewards_per_episode = np.zeros(Episodes)

  for a in range(Episodes):

    #Reset the environment to get the initial state
    env.reset()
    
    #Set the environment's current state to the desired starting position
    position = find_position_of_s(Custom_Map)
    env.env.s = position  # 'position' is the index of the desired starting position
    state = env.env.s

    terminated = False # True when agent falles in hole or reached goal
    truncated = False # True when actions > 200

    while(not terminated and not truncated):

      if rng.random() < epsilon: # epsilon is a parameter that controls the trade-off between exploration and exploitation

        #If True = the agent takes a random action
        #This is the exploration part

        action = env.action_space.sample() # actions: 0 = left, 1 = down, 2 = right, 3 = up

      else:

        # If false = the agent takes the action with the highest Q-value for the current state
        #This is the exploitation part, where the agent uses its current knowledge to take what it thinks is the best action.

        action = np.argmax(q[state, :])

      #new state if 0 and move right = 1, 1 if it reaches goal, terminated if fall or reach goal, truncated if actions over 200
      new_state, reward, terminated, truncated, _ = env.step(action)

      #Update Q values
      q[state, action] = q[state, action] + learning_rate_a * (reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action])

      #Q[state, action] is the current Q value
      #α is the learning rate (0 < α <= 1).
      #reward is the reward received for taking the action.
      #γ is the discount factor (0 <= γ < 1), which determines the importance of future rewards
      #max a′ Q(next_state,a′ ) is the estimated maximum future reward at the next state

      # Assign to new state after action to now its current state
      state = new_state
    
    #As epsilon decreases over time, the agent will explore less and exploit more, relying more on its accumulated knowledge
    epsilon = max(epsilon - epsilon_decay_rate, 0)

    #Helps stabilize Q values after exploration ends
    if(epsilon == 0):
      learning_rate_a = 0.0001

    if(reward == 1):
      rewards_per_episode[a] = 1
  
  #If finished close environment
  env.close()

  #Map rewards to a graph for easy understanding
  sum_rewards = np.zeros(Episodes)
  for t in range(Episodes):
    sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)]) #Show running sum of rewards for every 100 episodes
  plt.plot(sum_rewards)
  plt.savefig("Frozen_lake8x8.png")

#Run Main Function
if __name__ == "__main__":
  FrozenLake(maps["5x5"], 15000) # Load Map "5x5" and Do 15000 iterations(Episodes)