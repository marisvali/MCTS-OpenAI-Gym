from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import gymnasium as gym
from policy import Policy_Player_MCTS
from node import Node
from copy import deepcopy

# This runs the Monte Carlo Tree Search algorithm on any game.
# It is very slow (maybe 30 min for an episode, depending on the game).
# Rewards so far:
# 'CartPole-v0' = 200 (maximum)
# 'MountainCar-v0' = -200 (minimum)
# 'ALE/Breakout-v5' = 2 (close to minimum)

# GAME_NAME = 'MountainCar-v0'
# GAME_NAME = 'CartPole-v0'
GAME_NAME = 'ALE/Breakout-v5'

# env = gym.make(GAME_NAME, render_mode="human")
env = gym.make(GAME_NAME)

GAME_ACTIONS = env.action_space.n
GAME_OBS = env.observation_space.shape[0]

print('In the ' + GAME_NAME + ' environment there are: ' + str(GAME_ACTIONS) + ' possible actions.')
print('In the ' + GAME_NAME + ' environment the observation is composed of: ' + str(GAME_OBS) + ' values.')

env.reset()
env.close()

episodes = 10
rewards = []
moving_average = []

'''
Here we are experimenting with our implementation:
- we play a certain number of episodes of the game
- for deciding each move to play at each step, we will apply our MCTS algorithm
- we will collect and plot the rewards to check if the MCTS is actually working.
- For CartPole-v0, in particular, 200 is the maximum possible reward. 
'''

for e in range(episodes):

    reward_e = 0    
    # game = gym.make(GAME_NAME, render_mode="human", max_episode_steps=500)
    game = gym.make(GAME_NAME, max_episode_steps=500)
    observation = game.reset(seed=13) 
    done = False
    
    # new_game = deepcopy(game)
    # mytree = Node(new_game, False, 0, observation, 0, GAME_ACTIONS)
    new_game = gym.make(GAME_NAME, max_episode_steps=500)
    observation = new_game.reset(seed=13)
    mytree = Node(new_game, False, 0, observation, 0, GAME_ACTIONS)
    
    print('episode #' + str(e+1))
    
    while not done:
    
        mytree, action = Policy_Player_MCTS(mytree)
        print(f'#{action}', end='') # so we can see a step
        
        observation, reward, done, truncated, _ = game.step(action)  
        done = done or truncated

        reward_e = reward_e + reward
        
        # game.render() # uncomment this if you want to see your agent in action!
                
        if done:
            print('reward_e ' + str(reward_e))
            game.close()
            break
        
    rewards.append(reward_e)
    moving_average.append(np.mean(rewards[-100:]))
    
plt.plot(rewards)
plt.plot(moving_average)
plt.show()
print('moving average: ' + str(np.mean(rewards[-20:])))