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


env = environment(5)
check_env(env) #used to prepare env for training

#Randomness test
while True:
    action = random.randint(0,3)
    new_state,reward,done,info = env.step(action)
    print('new state\n',new_state)










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