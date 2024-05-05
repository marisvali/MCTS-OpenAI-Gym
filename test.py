'''
We will test our Monte Carlo Tree Search algorithm (MCTS) using an openAI gym environment named "CartPole".
You can read more information about the environment at this link:
https://www.gymlibrary.ml/environments/classic_control/cart_pole/

Feel free to change the environment with other as you like, changing the game name variable, 
but keep in mind that for this MCTS implementation both Actions and Observation must be Discrete. 
'''

# uncomment these lines below if you get a runtime error of gym package not found
# change the path value using your actual gym path using the 'pip show gym' command
import sys
path = "c:\\users\\vali\\appdata\\local\\programs\\python\\python310\\lib\\site-packages"
sys.path.append(path)

import gymnasium as gym

# GAME_NAME = 'CartPole-v0'
GAME_NAME = 'ALE/Breakout-v5'

env = gym.make(GAME_NAME)

GAME_ACTIONS = env.action_space.n
GAME_OBS = env.observation_space.shape[0]

print('In the ' + GAME_NAME + ' environment there are: ' + str(GAME_ACTIONS) + ' possible actions.')
print('In the ' + GAME_NAME + ' environment the observation is composed of: ' + str(GAME_OBS) + ' values.')

env.reset()
env.close()