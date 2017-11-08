import gym
import tflearn
import numpy as np


env = gym.make('MountainCar-v0')
env.reset()


for _ in range(1000):
    env.render()
    action = env.action_space.sample() # take a random action
    observation, reward, done, info = env.step(action)
    print(action)
    print(observation)
