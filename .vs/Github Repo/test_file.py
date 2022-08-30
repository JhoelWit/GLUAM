# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:14:24 2022

@author: ADAMS-LAB
"""

from environment import environment
from stable_baselines3.common.env_checker import check_env
import gym
import time
import random
import matplotlib.pyplot as plt
import numpy as np

env = environment(5) #Changed from 5
# check_env(env) #used to prepare env for training

#Randomness test
steps1 = 0
while True:
    action = random.randint(0,3)
    # action = 1
    start_time = time.time()
    new_state,reward,done,info = env.step(action)
    print('time for one step',time.time() - start_time)
    steps1+=1
    if steps1 % 100 == 0:
        env.reset()
        steps1 = 0

# for run in range(200):
#     action = random.randint(0,3)
#     new_state,reward,done,info = env.step(action)
    # print('new state\n',new_state)
# rewardls = []
# mean_rew = []

# for batches in range(20):
#     print(batches)
#     for steps in range(50):
#         action = random.randint(0,3)
#         new_state, reward, done, info = env.step(action)
#         rewardls.append(reward)
#     mean_rew.append(np.average(rewardls))
#     env.reset()

# steps = np.arange(len(mean_rew)) * 50
# plt.plot(steps,mean_rew)
# plt.xlabel('Steps')
# plt.ylabel('Mean Reward')
# plt.title('Random GRL Agent')
# plt.show()









"""
All actions for reference
LAnding
1.1 stay still - 0
1.2 land in empty port - 1
1.3 land in battery port - 2
1.4 move to empty hovering spot - 3

Takeoff
2.1 stay still - 0
2.2 takeoff - 1
2.3 move to battery port - 2
2.4 move from battery port - 3


"""