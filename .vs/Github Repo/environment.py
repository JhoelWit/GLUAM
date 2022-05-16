# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 18:48:45 2022

@author: ADAMS-LAB
"""
from os import environ

from req_cls import UAMs,ports
import airsim
from Action_Manager import ActionManager
from State_manager import StateManager
from gym import spaces
import gym
import copy
import time
import random
import numpy as np
from torch import tensor,Tensor,long

class environment(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, no_of_drones):
        super(environment,self).__init__()
        self.no_drones = no_of_drones
        self.current_drone = None
        self.state_manager = None
        self.action_space = spaces.Discrete(4) 
        self.client = airsim.MultirotorClient()
        # self.observation_space = spaces.Box(low=0,high=2,shape=(7,),dtype=np.float32) #[battery_capacity,empty_port,empty_hovering_spots,empty_battery_ports,status,collision_status,schedule ]
        self.observation_space = spaces.Dict(
            dict(
                vertiport_features = spaces.Box(low=0.0,high=2.0,shape=(10,2), dtype=np.float32),
                vertiport_edge = spaces.Box(low=0.0,high=9.0,shape=(2,90), dtype=np.float32),
                evtol_features = spaces.Box(low=0.0,high=4.0,shape=(5,4), dtype=np.float32),
                evtol_edge = spaces.Box(low=0.0,high=4.0,shape=(2,20), dtype=np.float32),
                next_drone_embedding = spaces.Box(low=0,high=2,shape=(7,),dtype=np.float32),
                mask = spaces.Box(low=0,high=1,shape=(4,),dtype=np.float32)
            ))
        self.drone_offsets = [[0,0,-2], [-6,0,-1], [2,-3,0], [-6,-4,-3], [3,0,-4]]
        # self.drone_offsets = [[-51.6,-11.2,-9], [-51.6,-14.3,-8], [2,-3,-8], [-0.1,0,-7], [3,0,-6], [-48.5,-11.10,-5], [-48.50,-14.40,-6.6], [-45.40,-11.30,-7.5], [-45.40,-14.40,-8.5], [-42.30,-11.20,-9.5], [-42.30,-14.50,-5.5], [-15.90,-63.20,-7], [-15.90,-60.10,-8], [-12.80,-63.20,-9], [-9.8,-63.20,-10], [-9.8,-60.10,-9.5], [-6.7,-63,-8.5]] #For the demo
        self._initialize() 

        #every step can be counted as 20 seconds (or whatever is better). not using actual times  

    def _initialize(self):
        self.near_misses = 0
        self.reward = 0
        self.env_time = 0
        self.tasks_completed = 0 #A task will be considered as a drone going to a destination and returning
        self.max_delay = 60
        self.total_delay = 0
        self.sampling_time = 0.1 #6 seconds per step regardless of drone times, subject to change
        self.all_drones = list()
        self.port = ports(self.no_drones)
        self.graph_prop = {'vertiport_features':{},'vertiport_edge':{},
                            'evtol_features':{},'evtol_edge':{},
                            'next_drone_embedding':{}, 'mask':{}}
        self.graph_prop['vertiport_edge'] = self.create_edge_connect(num_nodes=self.port.no_total)
        self.graph_prop['evtol_edge'] = self.create_edge_connect(num_nodes=self.no_drones)
        self.drone_feature_mat = np.zeros((self.no_drones,4)) #four features per drone
        for i in range(self.no_drones):
            drone_name = "Drone"+str(i)
            offset = self.drone_offsets[i]
            self.all_drones.append(UAMs(drone_name, offset))
            self.client.enableApiControl(True, drone_name)
            self.client.armDisarm(True, drone_name)
            self.takeoff(self.all_drones[i].drone_name,join=1)
        for i in self.all_drones:   
            x = self.client.getMultirotorState(i.drone_name).kinematics_estimated.position.x_val
            y = self.client.getMultirotorState(i.drone_name).kinematics_estimated.position.y_val
            z = self.client.getMultirotorState(i.drone_name).kinematics_estimated.position.z_val
            loc = [x,y,z]
            i.drone_locs[self.all_drones.index(i)] = loc
            i.current_location = loc
        
        self.state_manager = StateManager(self.port)
        self.action_manager = ActionManager(self.port)
        self.initial_schedule()
        # self.initial_setup()
        # print('environment initialized')

    def initial_schedule(self): 
        self.done = False
        for drone in self.all_drones:
            # choice = random.randint(0,1) #Controls whether a destination (0) or hover port (1) is picked
            choice = 1 #This will make sure the five drones start at the same spots, every time. Good for testing purposes 
            drone.assign_schedule(port = self.port,choice = choice)
            initial_des = drone.job_status['final_dest']
            if initial_des in self.port.hover_spots:
                drone.set_status("in-action", "in-air")
                drone.port_identification = {'type':'hover','port_no':self.port.hover_spots.index(initial_des)}
            else:
                drone.set_status("in-action", "in-destination")
            drone.job_status['initial_loc'] = drone.current_location
            self.move_position(drone.drone_name,initial_des,join=0)
        self.update_all()

    def initial_setup(self):
        """
        setup the initial state of the drones 
        one landed, two in destination and 1 will be in hoverspot

        Returns
        -------
        None.

        """
        hover_loc = self.port.get_empty_hover_status()
        final_pos = self.get_final_pos(hover_loc["position"], self.all_drones[1].offset)
        self.move_position(self.all_drones[1].drone_name, final_pos)
        self.all_drones[1].job_status["initial_loc"] = self.all_drones[1].current_location
        self.all_drones[1].job_status["final_dest"] = final_pos
        self.port.change_hover_spot_status(hover_loc["port_no"], True)
        self.all_drones[1].set_status("in-action", "in-air")
        #drone 2
        hover_loc = self.port.get_destination()
        self.move_position(self.all_drones[2].drone_name, hover_loc)
        self.all_drones[2].set_status("in-action", "in-destination")
        self.all_drones[2].job_status["initial_loc"] = self.all_drones[2].current_location
        self.all_drones[2].job_status["final_dest"] = hover_loc
        #drone 3
        hover_loc = self.port.get_destination()
        self.move_position(self.all_drones[3].drone_name, hover_loc,1)
        self.all_drones[3].set_status("in-action", "in-destination")
        self.all_drones[3].job_status["initial_loc"] = self.all_drones[3].current_location
        self.all_drones[3].job_status["final_dest"] = hover_loc
        
    def step(self,action):
        start_time = time.time()
        # print('stepping with action',action)
        self.Try_selecting_drone_from_Start()
        coded_action = self.action_manager.action_decode(self.current_drone, action)
        #Ideally, all movements should take place here, and ports that were previously unavailable should free up
        if coded_action["action"] == "land":
            new_position = coded_action["position"]
            self.complete_landing(self.current_drone.drone_name, new_position)
            old_position = self.current_drone.current_location
            reduce = self.current_drone.calculate_reduction(old_position,new_position) #Ditto
            self.current_drone.set_status("in-action", "in-port")
            self.current_drone.update_flight_time(reduce) #Ditto
            self.current_drone.job_status["initial_loc"] = self.current_drone.current_location
            self.current_drone.job_status["final_dest"] = new_position
            
        elif coded_action["action"] == "land-b":
            new_position = coded_action["position"]
            self.complete_landing(self.current_drone.drone_name, new_position)
            old_position = self.current_drone.current_location
            reduce = self.current_drone.calculate_reduction(old_position,new_position) #Ditto
            self.current_drone.set_status("in-action", "battery-port")
            self.current_drone.update_flight_time(reduce) #Ditto
            self.current_drone.job_status["initial_loc"] = self.current_drone.current_location
            self.current_drone.job_status["final_dest"] = new_position
            
        elif coded_action["action"] == "takeoff":
            new_position = coded_action["position"]
            self.complete_takeoff(self.current_drone.drone_name, new_position)
            old_position = self.current_drone.current_location
            reduce = self.current_drone.calculate_reduction(old_position,new_position) #Ditto
            self.current_drone.update_flight_time(reduce) #Ditto
            self.current_drone.set_status("in-action", "in-destination")
            self.current_drone.job_status["initial_loc"] = self.current_drone.current_location
            self.current_drone.job_status["final_dest"] = new_position
       
        elif coded_action["action"] == "hover":
            new_position = coded_action["position"]
            self.move_position(self.current_drone.drone_name, new_position,join=0)
            old_position = self.current_drone.current_location
            reduce = self.current_drone.calculate_reduction(old_position,new_position) #Ditto
            self.current_drone.update_flight_time(reduce) #Ditto
            self.current_drone.status = self.current_drone.all_states['in-air']
            self.current_drone.job_status["initial_loc"] = self.current_drone.current_location
            self.current_drone.job_status["final_dest"] = new_position            
        
        elif coded_action['action'] == 'takeoff-hover':
            new_position = coded_action["position"]
            self.complete_takeoff_hover(self.current_drone.drone_name,new_position)
            old_position = self.current_drone.current_location
            reduce = self.current_drone.calculate_reduction(old_position,new_position) #Ditto
            self.current_drone.update_flight_time(reduce) #Ditto
            self.current_drone.status = self.current_drone.all_states['in-air']
            self.current_drone.job_status["initial_loc"] = self.current_drone.current_location
            self.current_drone.job_status["final_dest"] = new_position

        elif coded_action['action'] == 'move-b':
            new_position = coded_action["position"]
            self.change_port(self.current_drone.drone_name, new_position)
            old_position = self.current_drone.current_location            
            reduce = self.current_drone.calculate_reduction(old_position,new_position) #Ditto
            self.current_drone.update_flight_time(reduce) #Ditto
            self.current_drone.job_status['final_dest'] = new_position
            self.current_drone.status_to_set = self.current_drone.all_states['battery-port']
            self.current_drone.job_status["initial_loc"] = self.current_drone.current_location
            self.current_drone.job_status["final_dest"] = new_position

        elif coded_action["action"] == "stay" or coded_action['action'] == 'stay-penalty':
            self.client.hoverAsync(self.current_drone.drone_name)
            reduce = 1
            self.current_drone.update_flight_time(reduce) #Ditto

        elif coded_action['action'] == 'continue':
            pass #Continuing on the path
        
        elif coded_action['action'] == 'deviate':
            pass #proof of concept, in later tests we will have the drones actually deviate
            # if self.current_drone.collision_status == 1:
            #     self.reward += 10
            # else:
            #     self.reward -= 5
            # print('deviating')
            # dev = random.randint(1,3)
            # self.current_drone.job_status['final_dest'][2] -= dev
            # pos = self.current_drone.job_status['final_dest']
            # self.client.moveToZAsync(z=pos[2],velocity=1,vehicle_name=self.current_drone.drone_name)
            # print('crisis averted')
            # self.current_drone.collision_status = 0 #Crisis averted
            # self.client.moveToPositionAsync(pos[0],pos[1],pos[2], velocity=1, vehicle_name=self.current_drone.drone_name).join()
                   
        elif coded_action['action'] == 'reward-penalty': #In the case that a drone lands on another drone, or does not properly avoid a collision
            pass
    
        else:
            new_position = coded_action["position"]
            self.change_port(self.current_drone.drone_name, new_position)
            old_position = self.current_drone.current_location
            reduce = self.current_drone.calculate_reduction(old_position,new_position) #Ditto
            self.current_drone.update_flight_time(reduce) #Ditto
            self.current_drone.job_status["initial_loc"] = self.current_drone.current_location
            self.current_drone.job_status["final_dest"] = new_position
            self.current_drone.set_status('in-action','in-port')
      
        self.update_all()
        # self.reward += self.calculate_reward(coded_action['action']) #adjusted the function, also going to just add reward values until we sort this out
        self.reward += self.calculate_simple_reward(coded_action['action'])
        self.select_next_drone()
        new_state = self._get_obs()
        self.env_time += self.sampling_time  #reduce self.sampling_time if the simulation is too fast
        # print('env time',self.env_time)
        reward = self.reward
        # print('reward is',reward)
        # self.debugg()
        if self.env_time >= self.max_delay: #Env resets after an hour (200 steps with self.sampling_time = 0.3)
            self.done = True
        done = self.done            
        info = {}
        self.step_time = time.time() - start_time
        return new_state,reward,done,info
        
    def debugg(self):
        a = [print(i.status) for i in self.all_drones]
    
    def complete_takeoff(self,drone_name, fly_port):
        self.client.moveToZAsync(self.drone_offsets[int(drone_name[-1])][-1],velocity=1,vehicle_name=drone_name).join()
        # self.client.takeoffAsync(vehicle_name=drone_name).join() #Stopped using takeoffAsync due to no offsets and constant collisions between drones
        self.move_position(drone_name, fly_port,join=0)

    def complete_takeoff_hover(self,drone_name, fly_port):
        self.client.moveToZAsync(self.drone_offsets[int(drone_name[-1])][-1],velocity=1,vehicle_name=drone_name).join()
        # self.client.takeoffAsync(vehicle_name=drone_name).join()
        self.move_position(drone_name, fly_port,join=1)
        
    def move_position(self, drone_name, position,join=0):
        if join == 0:
            self.client.moveToPositionAsync(position[0],position[1],self.drone_offsets[int(drone_name[-1])][-1], velocity=1, vehicle_name=drone_name)
        else:
            self.client.moveToPositionAsync(position[0],position[1],self.drone_offsets[int(drone_name[-1])][-1], velocity=1, vehicle_name=drone_name).join()
    
    def complete_landing(self, drone_name, location):  
        self.move_position(drone_name, location,join=1)
        self.client.landAsync(vehicle_name=drone_name).join()
        
    def select_next_drone(self):
        failed_attempts = 0
        if not self.current_drone:
            self.current_drone = self.all_drones[0]
            new_drone_no = 0
        else:
            old_drone_no = self.all_drones.index(self.current_drone)
            new_drone_no = old_drone_no + 1
        if new_drone_no < self.no_drones:
            self.current_drone = self.all_drones[new_drone_no]
            collision = self.client.simGetCollisionInfo(self.current_drone.drone_name)
            while self.current_drone.status == 4 or (collision.has_collided == True and collision.object_name[:-1] == 'Drone'):  #Might need to change conditionals for bigger environments with more obstacles
                time.sleep(1)
                self.update_all()
                collision = self.client.simGetCollisionInfo(self.current_drone.drone_name)
                failed_attempts += 1
                if failed_attempts > 10: #Failsafe
                    self.client.reset()
                    time.sleep(3)
                    self.enable_control(False)
                    self.enable_control(True)
                    self.port.reset_ports()
                    for i in range(self.no_drones):
                        self.takeoff(self.all_drones[i].drone_name,join=1)
                    self.initial_schedule()
                    return
        else: #All drones have taken an action, resetting
            self.current_drone = self.all_drones[0]
        self.update_all()
        # print(["status of current drone",self.current_drone.drone_name,'is',self.current_drone.status])
        
    def Try_selecting_drone_from_Start(self):
        """
        This function picks the first drone to take an action, or if the first drone is picked the function will wait until the drone is in-airspace

        Returns
        -------
        None.

        """
        failed_attempts = 0
        if not self.current_drone: #Means this is the first step
            self.current_drone = self.all_drones[0]
            self.update_all()
            collision = self.client.simGetCollisionInfo(self.current_drone.drone_name)
            while self.current_drone.status == 4 or (collision.has_collided == True and collision.object_name[:-1] == 'Drone'):   
                time.sleep(1)
                self.update_all()
                collision = self.client.simGetCollisionInfo(self.current_drone.drone_name)
                failed_attempts += 1
                if failed_attempts > 10: #Failsafe
                    self.client.reset()
                    time.sleep(3)
                    self.enable_control(False)
                    self.enable_control(True)
                    self.port.reset_ports()
                    for i in range(self.no_drones):
                        self.takeoff(self.all_drones[i].drone_name,join=1)
                    self.initial_schedule()
                    return
        else:
            collision = self.client.simGetCollisionInfo(self.current_drone.drone_name)
            while self.current_drone.status == 4 or (collision.has_collided == True and collision.object_name[:-1] == 'Drone'):   
                time.sleep(1)
                self.update_all()
                collision = self.client.simGetCollisionInfo(self.current_drone.drone_name)
                failed_attempts += 1
                if failed_attempts > 10: #Failsafe
                    self.client.reset()
                    time.sleep(3)
                    self.enable_control(False)
                    self.enable_control(True)
                    self.port.reset_ports()
                    for i in range(self.no_drones):
                        self.takeoff(self.all_drones[i].drone_name,join=1)
                    self.initial_schedule()
                    return
    
    def enable_control(self, control):
        for i in range(self.no_drones):
            self.client.enableApiControl(control, self.all_drones[i].drone_name)
        for i in range(self.no_drones):
             self.client.armDisarm(control, self.all_drones[i].drone_name)

    def _get_obs(self):
        states = self.state_manager.get_obs(self.current_drone,self.graph_prop) #Used for GRL agent
        # states = self.state_manager.get_obs(self.current_drone) #Used for baseline DRL agent

        return states
    
    def calculate_simple_reward(self,action):
        drone = self.current_drone
        Dmax = self.max_delay
        tasks = self.tasks_completed
        xtra_penalty = 0
        a1 = 0
        lambda_bat = 0
        if  action == "takeoff" or action == 'takeoff-hover':
            D = max(0,self.current_drone.upcoming_schedule['time'] - self.current_drone.upcoming_schedule["takeoff-time"]) #Replaced self.env_time with drone time from time updates
        elif action == "land-b" or action == "land" or action == "hover" or action == 'move' or action == 'stay-penalty':
            D = max(0,self.current_drone.upcoming_schedule['time'] - self.current_drone.upcoming_schedule["landing-time"]) #Ditto
        else:
            D = max(0,self.current_drone.upcoming_schedule['time'] - self.current_drone.upcoming_schedule["landing-time"])
        if drone.status == drone.all_states['in-action']:
            if drone.collision_status == 1 and action == 'deviate':
                self.near_misses += 1
                a1 = 0
                drone.collision_status = 0
            elif drone.collision_status == 1:
                a1 = 1
        if action == 'takeoff-hover' or action == 'stay-penalty': #These actions mean the agent tried to land a drone while the availability of the ports was 0, in this case it should have moved the drones to hover
            xtra_penalty = 3
        if drone.battery_state == drone.all_battery_states['critical']:
            lambda_bat = 1
        if D > Dmax:
            D = Dmax
        self.total_delay += D
        simple_reward = -((D / Dmax) + (2 * lambda_bat) + xtra_penalty + (4 * a1) + 10*np.exp(-0.1*tasks)) #Going to add another term for tasks penalty -- Essentially rewards the robot for the amount of tasks it can complete
        # print('simple reward',simple_reward)
        return simple_reward

    def get_reward_vars(self,action):
        drone = self.current_drone
        a1,a2,a3,lambda_takeoff = 0,0,0,0
        if  action == "takeoff":
            delay = max(0,self.current_drone.upcoming_schedule['time'] - self.current_drone.upcoming_schedule["takeoff-time"]) #Replaced self.env_time with drone time from time updates
            a1 = 1
            if drone.battery_state == 0:
                lambda_takeoff  = -1
            else:
                lambda_takeoff = 1
        elif action == "land-b" or action == "land" or action == "hover":
            delay = max(0,self.current_drone.upcoming_schedule['time'] - self.current_drone.upcoming_schedule["landing-time"]) #Ditto
            a2 = 1
        elif action == 'deviate' or action == 'continue': #For uncertainties
            delay = max(0,self.current_drone.upcoming_schedule['time'] - self.current_drone.upcoming_schedule["takeoff-time"]) #Replaced self.env_time with drone time from time updates
            a3 = 1
        else:
            time = self.current_drone.upcoming_schedule['time']
            # print(time)
            if time > 5000:
                self.done = True
            T = time
           
            return 0,0,0,0,0,0 #no reward
        self.total_delay += (delay/60)
        T = delay

        if delay > 5000:
            self.done = True
        #picking a coefficients based on the drone status
        if delay <= (0.5*self.sampling_time):
            beta = delay/ (60*10)
        else:
            beta = 3


        return T,a1,a2,a3,lambda_takeoff,beta



    def calculate_reward(self,action):
        # B = i/10
        # min_b2 = -B**2
        # R = -(i/N_of_uams) + (j * math.exp(min_b2/10))
        '''
        Reward V3:
        ---------

        Inputs: 

        T = delay time of vehicle n
        α1,2,3 = takeoff, landing, stay coefficients 
        β = Timing coefficient[-1,1]
        σ = Landing port selection (0,1)
            1 = chose the right port 
            0 = otherwise
        ϒ = Stay coefficient
        λ = battery condition(-1,1)
            -1 = less than recommended 
            1 = otherwise
        ε = Early timing (-1,1)
            -1 = early
            1 = otherwise
        
        Returns:

        reward
        '''
        #calculating delay time:

        T,a1,a2,a3,lambda_takeoff,beta = self.get_reward_vars(action)
        reward = -(T/60) + (a1*(lambda_takeoff * np.exp(-beta**2))) + (a2 * np.exp(-beta)) #Still gotta work on that third term
        # print(["self.current_drone : ", self.current_drone.drone_name, " reward = ", reward, " delay = ", T])

        return reward
    
    def update_all(self):
        """
        This function is called during every step, drones should be updated with the current position and status

        Returns
        -------
        None.

        """
        self.tasks_completed = 0
        for i in self.all_drones:    
            x = self.client.getMultirotorState(i.drone_name).kinematics_estimated.position.x_val
            y = self.client.getMultirotorState(i.drone_name).kinematics_estimated.position.y_val
            z = self.client.getMultirotorState(i.drone_name).kinematics_estimated.position.z_val
            loc = [x,y,z]
            i.update_time(loc,self.sampling_time)
            i.current_location = loc
            i.drone_locs[self.all_drones.index(i)] = loc #Change drone_locs size if changing number of drones
            i.update(loc,self.client,self.port) 
            i.collision_avoidance()
            i.get_state_status()
            self.tasks_completed += i.tasks_completed
            # print('drone',i.drone_name,'schedule status:',i.schedule_status)
            self.drone_feature_mat[self.all_drones.index(i)] = [i.battery_state, i.status, i.collision_status, i.schedule_status] #battery state, current mode, and collision possibility
        self.port.update_all()
        self.graph_prop['vertiport_features'] = self.port.feature_mat
        self.graph_prop['evtol_features'] = self.drone_feature_mat
    


    def reset(self):
        self.client.reset()
        time.sleep(3)
        self.enable_control(False)
        self._initialize()
        self.current_drone = self.all_drones[0]
        return self.state_manager.get_obs(self.current_drone,self.graph_prop)
        # return self.state_manager.get_obs(self.current_drone)
    
    
    def change_port(self, drone_name, new_port):
        self.client.moveToZAsync(self.drone_offsets[int(drone_name[-1])][-1],velocity=1,vehicle_name=drone_name).join()
        self.client.moveToPositionAsync(new_port[0],new_port[1],self.drone_offsets[int(drone_name[-1])][-1], velocity=1, vehicle_name=drone_name).join()
        self.client.landAsync(vehicle_name=drone_name).join()
    
    

    def takeoff(self, drone_name, join=0):
        if join == 0:
            self.client.moveToZAsync(self.drone_offsets[int(drone_name[-1])][-1],velocity=1,vehicle_name=drone_name)
        else:
            self.client.moveToZAsync(self.drone_offsets[int(drone_name[-1])][-1],velocity=1,vehicle_name=drone_name).join()
    
    def get_final_pos(self,port, offset):
        return [port[0] + offset[0] , port[1] + offset[1], offset[2]] #Using offset to keep every drone on different altitudes

    def create_edge_connect(self,num_nodes=5,adj_mat=None,direct_connect=0):
        if direct_connect == 0: #undirected graph, every node is connected and shares info with each other
            k = num_nodes-1
            num_edges = k*num_nodes
            blank_edge = np.ones( (2 , num_edges) )
            top_index = 0
            bot_index = np.arange(num_nodes)
            index = 1
            for i in range(num_nodes):
                blank_edge[0][k*i:k*index] = top_index
                blank_edge[1][k*i:k*index] = np.delete(bot_index,top_index)
                index+=1
                top_index+=1
        elif direct_connect == 1: #directed graph, in which case we need the adjacency matrix to create the edge list tensor
            blank_edge = np.array([]).reshape(2,0)
            for i in range(adj_mat.shape[0]):
                for j in range(adj_mat.shape[1]):
                    if adj_mat[i,j] == 1:
                        blank_edge = np.concatenate((blank_edge,np.array([i,j]).reshape(2,1)),axis=1)
        # return tensor(blank_edge,dtype=long) #gym.spaces don't want tensors, they tensorfy the output anyway
        return blank_edge
    
    def close(self): #This is never being used, unless we use some high power computer that runs 10_000_000 steps in an unconceivable amount of time...
        pass

