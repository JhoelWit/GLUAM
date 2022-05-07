# -*- coding: utf-8 -*-
"""
Created on Sat Mar  25 12:51:19 2022

@author: praji
"""


# import setup_path 
import airsim
import numpy as np
import os
import tempfile
import pprint
import time
import cv2
import gym
# import keras
import random
import time
import math


# from stable_baselines import A2C
# from stable_baselines.common.evaluation import evaluate_policy
# from stable_baselines import results_plotter
# from stable_baselines.bench import Monitor
# from stable_baselines.results_plotter import load_results, ts2xy
# from stable_baselines.common.noise import AdaptiveParamNoiseSpec
# from stable_baselines.common.callbacks import BaseCallback
from gym import spaces
# from tensorflow.python.framework.ops import disable_eager_execution
# import matplotlib.pyplot as plt

# disable_eager_execution()

class airsim_env(gym.Env):
    def __init__(self,no_of_drones):

        self.client = airsim.MultirotorClient()
        self.start_time = time.time()
        #self.client.reset()
        self.no_of_drones = no_of_drones
        self.ports = [[0,0], [-3,0], [-2,3]]
        self.offsets = [[0,0], [-6,0], [-4,-4], [-6,-4] ]
        self.fake_spots = [[13,4], [10,1], [-10,1]]
        self.hover_spots = [[13,4], [10,1], [-10,1]]
        self.all_data_store = []
        self.enable_control()
        self.client.confirmConnection()
        #for i in range(1,no_of_drones+1):
            #self.takeoff(i)
        # self.local_timesteps = 0
        # self.action_space = spaces.Box(low = 0, high = 1, shape=(4,))
        # self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(4,))
        #create rotor connection start simulation

    def step(self):
        
        #state1 = self.client.getMultirotorState(vehicle_name="Drone1")
       # state2 = self.client.getMultirotorState(vehicle_name="Drone2")

        #new_state = np.array(self._getobs())

        f1 = self.client.moveToPositionAsync(1, 1, 0.5, 5, vehicle_name="Drone1")
        f2 = self.client.moveToPositionAsync(2, 2, 0.5, 5, vehicle_name="Drone2")
        f3 = self.client.moveToPositionAsync(3, 3, -0.5, 5, vehicle_name="Drone3")
        f4 = self.client.moveToPositionAsync(4, 4, -0.5, 5, vehicle_name="Drone4")

        print(self.client.getMultirotorState(vehicle_name="Drone1"))
        #return new_state,reward,done,info
        
    def temp_action(self, drone_no, action):
        pass

    def _getobs(self):
        state1 = self.client.getMultirotorState(vehicle_name="Drone1")
        state2 = self.client.getMultirotorState(vehicle_name="Drone2")

        state = self.client.getMultirotorState()

    def Check(self):
        self._getobs()
        
    def takeoff(self, drone_no):
        self.client.takeoffAsync(vehicle_name="Drone"+str(drone_no)).join()
    def land(self, drone_no):
        self.client.landAsync(vehicle_name="Drone"+str(drone_no)).join()
        
    def movetoPort(self,drone_no, port_no):
        self.client.moveToPositionAsync(self.ports[port_no][0] + self.offsets[drone_no-1][1],self.ports[port_no][0] + self.offsets[drone_no-1][1],-2, 1, vehicle_name="Drone"+str(drone_no)).join()

    
    def movetohover(self,drone_no, port_no):
        self.client.moveToPositionAsync(self.ports[port_no][0] + self.offsets[drone_no-1][1],self.ports[port_no][0] + self.offsets[drone_no-1][1],-2, 1, vehicle_name="Drone"+str(drone_no)).join()

        
    def movetofakespot(self,drone_no, port_no):
        self.client.moveToPositionAsync(self.ports[port_no][0] + self.offsets[drone_no-1][1],self.ports[port_no][0] + self.offsets[drone_no-1][1],-2, 1, vehicle_name="Drone"+str(drone_no)).join()

        
    def complete_landing(self, drone_no, port_no):
        print(drone_no)
        print(self.ports[port_no])
        print(self.offsets[drone_no-1])
        print(self.ports[port_no][0] + self.offsets[drone_no-1][0],self.ports[port_no][1] + self.offsets[drone_no-1][1])
        self.client.moveToPositionAsync(self.ports[port_no][0] + self.offsets[drone_no-1][0],self.ports[port_no][1] + self.offsets[drone_no-1][1],-2, 1, vehicle_name="Drone"+str(drone_no)).join()
        self.client.landAsync(vehicle_name="Drone"+str(drone_no))
        
    def move_position(self, drone_no, position):
        self.client.moveToPositionAsync(position[0] + self.offsets[drone_no-1][1],position[1] + self.offsets[drone_no-1][1],position[2], 1, vehicle_name="Drone"+str(drone_no))

    def complete_takeoff(self,drone_no, fly_port):
        self.client.takeoffAsync(vehicle_name="Drone"+str(drone_no)).join()
        self.move_position(drone_no,[ self.fake_spots[fly_port][0],self.fake_spots[fly_port][1], -10] )
        
    def enable_control(self):
        self.client.enableApiControl(True, "Drone1")
        self.client.enableApiControl(True, "Drone2")
        self.client.enableApiControl(True, "Drone3")
        self.client.enableApiControl(True, "Drone4")
        self.client.armDisarm(True, "Drone1")
        self.client.armDisarm(True, "Drone2")
        self.client.armDisarm(True, "Drone3")
        self.client.armDisarm(True, "Drone4")
        
    def reset(self):
        self.client.reset()
        self.client = airsim.MultirotorClient()
        self.client.enableApiControl(True)
        self.start_time = time.time()
        self.client.armDisarm(True)
        self.client.confirmConnection()
        time.sleep(1)
        self.local_timesteps = 0
        self.no_collisions = 0
        new_state = self._getobs()
        return np.array(new_state).reshape(4,)



env = airsim_env(4)

env.takeoff(2)
env.complete_landing(2, 2)
env.takeoff(3)
env.complete_landing(3, 1)
env.takeoff(4)
env.movetohover(4, 1)
env.complete_takeoff(1, 2)
env.complete_landing(4, 0)
env.movetohover(1, 2)
env.complete_takeoff(2, 1)
env.complete_landing(1, 2)
env.complete_takeoff(3, 0)
env.complete_takeoff(4, 1)
# env.step()