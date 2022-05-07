# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:14:24 2022

@author: ADAMS-LAB
"""

from environment import environment
import gym


env = environment(4)
new_state,reward,done,info = env.step(2)
print('State\n',new_state,'\nreward\n',reward,'\ndone\n',done,'\ninfo\n',info)