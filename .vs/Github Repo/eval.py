import numpy as np
import gym
import torch
# from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
import pickle
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from environment import environment
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from GL_Policy import CustomGLPolicy, CustomBaselinePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import matplotlib.pyplot as plt
import random
import os
import csv


# with open("test_results_scitech_1_percent_noise.csv", mode="w", newline="") as csvfile:
#     writer = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(["Agent", "Reward", "Collisions", "Delay", "Good Takeoffs", "Good Landings", "Battery", "Step Time", "Problem"])


#     env = environment(no_of_drones=4, type="graph", test=True, noise=False)

#     ep_len = 1440
#     test_eps = 50
#     model = PPO.load('ATC_Model/683/ATC_GRL_Model_160000_steps') #Noise
#     obs = env.reset()
#     for i in range(1,ep_len*test_eps+1):
#         action, _ = model.predict(obs,deterministic=True)
#         if i % ep_len == 0 and i > 0:
#             reward, collisions, total_delay, good_takeoffs, good_landings, avg_battery, step_time = env.step(action)
#             writer.writerow(["GRL_1noise", reward, collisions, total_delay / 60, good_takeoffs, good_landings, avg_battery, step_time, "UAM-VSM"])
#             print("Iteration", i // ep_len, "stats:", [reward, collisions, total_delay / 60, good_takeoffs, good_landings, avg_battery, step_time] )
#             obs = env.reset()
#         else:
#             obs, reward, done, info = env.step(action)

#     del model
#     del env

    # env = environment(no_of_drones=4, type="graph", test=True, noise=True)

    # ep_len = 1440
    # test_eps = 50
    # model = PPO.load('ATC_Model/683/ATC_GRL_Model_160000_steps')
    # obs = env.reset()
    # for i in range(1,ep_len*test_eps+1):
    #     action, _ = model.predict(obs,deterministic=True)
    #     if i % ep_len == 0 and i > 0:
    #         reward, collisions, total_delay, good_takeoffs, good_landings, avg_battery, step_time = env.step(action)
    #         writer.writerow(["GRL_noise", reward, collisions, total_delay / 60, good_takeoffs, good_landings, avg_battery, step_time, "UAM-VSM"])
    #         print("Iteration", i // ep_len, "stats:", [reward, collisions, total_delay / 60, good_takeoffs, good_landings, avg_battery, step_time] )
    #         obs = env.reset()
    #     else:
    #         obs, reward, done, info = env.step(action)
    
    # del model
    # del env

    # env = environment(no_of_drones=4, type="regular", test=True)

    # ep_len = 1440
    # test_eps = 50
    # obs = env.reset()
    # for i in range(1,ep_len*test_eps+1):
    #     action = random.randint(0,10)
    #     if i % ep_len == 0 and i > 0:
    #         reward, collisions, total_delay, good_takeoffs, good_landings, avg_battery, step_time = env.step(action)
    #         writer.writerow(["Random_no_noise", reward, collisions, total_delay / 60, good_takeoffs, good_landings, avg_battery, step_time, "UAM-VSM"])
    #         print("Iteration", i // ep_len, "stats:", [reward, collisions, total_delay / 60, good_takeoffs, good_landings, avg_battery, step_time] )
    #         obs = env.reset()
    #     else:
    #         obs, reward, done, info = env.step(action)
    # del env

    # env = environment(no_of_drones=4, type="regular", test="FCFS")

    # ep_len = 1440
    # test_eps = 50
    # obs = env.reset()
    # for i in range(1,ep_len*test_eps+1):
    #     action = None
    #     if i % ep_len == 0 and i > 0:
    #         reward, collisions, total_delay, good_takeoffs, good_landings, avg_battery, step_time = env.step(action)
    #         writer.writerow(["FCFS_no_noise", reward, collisions, total_delay / 60, good_takeoffs, good_landings, avg_battery, step_time, "UAM-VSM"])
    #         print("Iteration", i // ep_len, "stats:", [reward, collisions, total_delay / 60, good_takeoffs, good_landings, avg_battery, step_time] )
    #         obs = env.reset()
    #     else:
    #         obs, reward, done, info = env.step(action)
    
    # del env

    # """   With noise ----------------------------------------------------------------------"""    
    # env = environment(no_of_drones=4, type="graph", test=True, noise=True)

    # ep_len = 1440
    # test_eps = 50
    # model = PPO.load('ATC_Model/683/ATC_GRL_Model_160000_steps')
    # obs = env.reset()
    # for i in range(1,ep_len*test_eps+1):
    #     action, _ = model.predict(obs,deterministic=True)
    #     if i % ep_len == 0 and i > 0:
    #         reward, collisions, total_delay, good_takeoffs, good_landings, avg_battery, step_time = env.step(action)
    #         writer.writerow(["GRL_noise", reward, collisions, total_delay / 60, good_takeoffs, good_landings, avg_battery, step_time, "UAM-VSM"])
    #         print("Iteration", i // ep_len, "stats:", [reward, collisions, total_delay / 60, good_takeoffs, good_landings, avg_battery, step_time] )
    #         obs = env.reset()
    #     else:
    #         obs, reward, done, info = env.step(action)
    
    # del model
    # del env

    # env = environment(no_of_drones=4, type="regular", test=True, noise=True)

    # ep_len = 1440
    # test_eps = 50
    # obs = env.reset()
    # for i in range(1,ep_len*test_eps+1):
    #     action = random.randint(0,10)
    #     if i % ep_len == 0 and i > 0:
    #         reward, collisions, total_delay, good_takeoffs, good_landings, avg_battery, step_time = env.step(action)
    #         writer.writerow(["Random_noise", reward, collisions, total_delay / 60, good_takeoffs, good_landings, avg_battery, step_time, "UAM-VSM"])
    #         print("Iteration", i // ep_len, "stats:", [reward, collisions, total_delay / 60, good_takeoffs, good_landings, avg_battery, step_time] )
    #         obs = env.reset()
    #     else:
    #         obs, reward, done, info = env.step(action)
    # del env

    # env = environment(no_of_drones=4, type="regular", test="FCFS", noise=True)

    # ep_len = 1440
    # test_eps = 50
    # obs = env.reset()
    # for i in range(1,ep_len*test_eps+1):
    #     action = None
    #     if i % ep_len == 0 and i > 0:
    #         reward, collisions, total_delay, good_takeoffs, good_landings, avg_battery, step_time = env.step(action)
    #         writer.writerow(["FCFS_noise", reward, collisions, total_delay / 60, good_takeoffs, good_landings, avg_battery, step_time, "UAM-VSM"])
    #         print("Iteration", i // ep_len, "stats:", [reward, collisions, total_delay / 60, good_takeoffs, good_landings, avg_battery, step_time] )
    #         obs = env.reset()
    #     else:
    #         obs, reward, done, info = env.step(action)
    
    # del env


    #Action Distribution
with open("test_results_scitech_action_distribution.csv", mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["Agent", "Case", "Step", "Action"])


    env = environment(no_of_drones=4, type="graph", test=False, noise=False)

    ep_len = 1440
    test_eps = 5
    model = PPO.load('ATC_Model/683/ATC_GRL_Model_160000_steps') #Noise
    obs = env.reset()
    for i in range(1,ep_len*test_eps+1):
        action, _ = model.predict(obs,deterministic=True)
        if i % ep_len == 0 and i > 0:
            obs = env.reset()
        else:
            obs, reward, done, info = env.step(action)
            writer.writerow(["GRL", "Case 1", i, info["action"]])

    del model
    del env

    env = environment(no_of_drones=4, type="graph", test=False, noise=True)

    ep_len = 1440
    test_eps = 5
    model = PPO.load('ATC_Model/683/ATC_GRL_Model_160000_steps') #Noise
    obs = env.reset()
    for i in range(1,ep_len*test_eps+1):
        
        action, _ = model.predict(obs,deterministic=True)
        if i % ep_len == 0 and i > 0:
            obs = env.reset()
        else:
            obs, reward, done, info = env.step(action)
            writer.writerow(["GRL", "Case 2", i, info["action"]])

    del model
    del env
