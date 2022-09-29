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


env = environment(no_of_drones=4, type="graph", test=False)
# check_env(env) #used to prepare env for training
# actions1 = {'Drone0':1,'Drone1':1,'Drone2':2,'Drone3':1}
# actions2 = {'Drone0':3,'Drone1':1,'Drone2':2,'Drone3':1}
# actions3 = {'Drone0':0,'Drone1':0,'Drone2':0,'Drone3':0}


#Randomness test
i = 0
while True:
    action = random.randint(0,10)
    # action = 0
    # new_state,reward,done,info = env.step(action)
    env.step(action)
    i += 1
    if i % 1440 == 0:
        env.reset()









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