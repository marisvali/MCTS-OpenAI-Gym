import gymnasium as gym

# This just runs a game, plays random actions and displays it.

# GAME_NAME = 'MountainCar-v0'
# GAME_NAME = 'CartPole-v0'
GAME_NAME = 'ALE/Breakout-v5'

env = gym.make(GAME_NAME, render_mode="human")
# env = gym.make(GAME_NAME)

GAME_ACTIONS = env.action_space.n
GAME_OBS = env.observation_space.shape[0]

print('In the ' + GAME_NAME + ' environment there are: ' + str(GAME_ACTIONS) + ' possible actions.')
print('In the ' + GAME_NAME + ' environment the observation is composed of: ' + str(GAME_OBS) + ' values.')

env.reset()
env.close()

reward_e = 0
game = gym.make(GAME_NAME, render_mode="human")
observation = game.reset() 
done = False

while True:
    print('#', end='') # so we can see a step
    
    action = game.action_space.sample()
    observation, reward, done, truncated, _ = game.step(action)  
    done = done or truncated

    reward_e = reward_e + reward
    
    # game.render() # uncomment this if you want to see your agent in action!
    if done:
        print('reward_e ' + str(reward_e))
        game.close()
        break
