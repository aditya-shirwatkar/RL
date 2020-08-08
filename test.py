# from five_link_env import *
# import time
# env = FiveLink2dEnv()

import gym
import gym_custom_envs
import time
env = gym.make('FiveLink-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation.shape)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        time.sleep(0.01)