# -*- coding: utf-8 -*-
"""
Created on Sat Mar  25 12:51:19 2022

@author: praji
"""


import setup_path 
import airsim
import numpy as np
import os
import tempfile
import pprint
import time
import cv2
import gym
import keras
import random
import time
import math


from stable_baselines import A2C
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback
from gym import spaces
from tensorflow.python.framework.ops import disable_eager_execution
import matplotlib.pyplot as plt

disable_eager_execution()

class airsim_env(gym.Env):
    def __init__(self,ip_address):


        self.ip_address = ip_address
        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.start_time = time.time()
        self.drone.reset()
        self.all_data_store = []
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.confirmConnection()
        f1 = self.drone.takeoffAsync(vehicle_name="Drone1")
        f2 = self.drone.takeoffAsync(vehicle_name="Drone2")
        f1.join()
        f2.join()
        self.local_timesteps = 0
        self.action_space = spaces.Box(low = 0, high = 1, shape=(4,))
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(4,))
        #create rotor connection start simulation

    def step(self,action):
        
        state1 = self.drone.getMultirotorState(vehicle_name="Drone1")
        state2 = self.drone.getMultirotorState(vehicle_name="Drone2")

        new_state = np.array(self._getobs())


        return new_state,reward,done,info

    def _getobs(self):
        state1 = self.drone.getMultirotorState(vehicle_name="Drone1")
        state2 = self.drone.getMultirotorState(vehicle_name="Drone2")

        state = self.drone.getMultirotorState()
        #pos_x = state.kinematics_estimated.position.x_val
        #pos_y = state.kinematics_estimated.position.y_val
       # pos_z = state.kinematics_estimated.position.z_val
       # ori_w = state.kinematics_estimated.orientation.w_val
       # state_space = [pos_x,pos_y,pos_z,ori_w]
        return np.array(state).reshape(4,)

    def Check(self):
        self._getobs()
        
    def reset(self):
        self.drone.reset()
        self.drone = airsim.MultirotorClient(ip=self.ip_address)
        self.drone.enableApiControl(True)
        self.start_time = time.time()
        self.drone.armDisarm(True)
        self.drone.confirmConnection()
        time.sleep(1)
        self.local_timesteps = 0
        self.no_collisions = 0
        new_state = self._getobs()
        return np.array(new_state).reshape(4,)



