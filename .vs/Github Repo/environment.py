# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 18:48:45 2022

@author: prajit
"""
from os import environ
from req_cls import UAMs,ports
import airsim
from Action_Manager import ActionManager, GL_ActionManager
from State_manager import StateManager, GL_StateManager
from gym import spaces
import gym
import copy
import time
import random
import numpy as np
from sympy.geometry import Segment3D, Segment2D

class environment(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, no_of_drones, type):
        super(environment,self).__init__()
        self.no_drones = no_of_drones
        self.current_drone = None
        self.state_manager = None
        self.total_timesteps = 0
        self.clock_speed = 30
        self.start_time = time.time()                   #environment times will be in seconds
        self.env_time = 0
        cont_bound = np.finfo(np.float32).max
        if type == "regular":
            self.action_space = spaces.Discrete(13) 
            # self.observation_space = spaces.MultiDiscrete([3,3,3,3,2,2]) # Can't use discrete with location values. 
            self.observation_space = spaces.Box(low=-cont_bound, high=cont_bound, shape=(8,), dtype=np.float32) #[battery_capacity,empty_port,empty_hovering_spots,empty_battery_ports,status,schedule, x, y, z ]
        elif type == "graph":
            self.action_space = spaces.Discrete(13) 
            self.observation_space = spaces.Dict(
            dict(
                vertiport_features = spaces.Box(low=-cont_bound,high=cont_bound,shape=(7,4), dtype=np.float32),
                vertiport_edge = spaces.Box(low=0.0,high=9.0,shape=(2,42), dtype=np.float32),
                evtol_features = spaces.Box(low=-cont_bound,high=cont_bound,shape=(4,5), dtype=np.float32),
                evtol_edge = spaces.Box(low=0.0,high=4.0,shape=(2,12), dtype=np.float32),
                next_drone_embedding = spaces.Box(low=-cont_bound,high=cont_bound,shape=(8,),dtype=np.float32),
                mask = spaces.Box(low=0,high=1,shape=(13,),dtype=np.float32)
            ))
        self.type = type
        self.client = airsim.MultirotorClient()
        self.drone_offsets = [[0,0,-2], [-6,0,-1], [2,-3,0], [-6,-4,-3], [3,0,-4]]
        self._initialize() 

        #every step can be counted as 20 seconds (or whatever is better). not using actual times

    def _initialize(self):

        self.env_time = (time.time() - self.start_time) *self.clock_speed
        self.total_timesteps = 0
        self.all_drones = list()
        self.port = ports(self.no_drones)
        self.graph_prop = {'vertiport_features':{},'vertiport_edge':{},
                            'evtol_features':{},'evtol_edge':{},
                            'next_drone_embedding':{}, 'mask':{}}
        self.graph_prop['vertiport_edge'] = self.create_edge_connect(num_nodes=self.port.no_total)
        self.graph_prop['evtol_edge'] = self.create_edge_connect(num_nodes=self.no_drones)
        self.drone_feature_mat = np.zeros((self.no_drones,5)) #five features per drone
        for i in range(self.no_drones):
            drone_name = "Drone"+str(i)
            offset = self.drone_offsets[i]
            self.all_drones.append(UAMs(drone_name, offset))
            self.client.enableApiControl(True, drone_name)
            self.client.armDisarm(True, drone_name)
            self.takeoff(self.all_drones[i].drone_name)
            # print('takeoff',drone_name)
        for i in self.all_drones:   
            x = self.client.getMultirotorState(i.drone_name).kinematics_estimated.position.x_val
            y = self.client.getMultirotorState(i.drone_name).kinematics_estimated.position.y_val
            z = self.client.getMultirotorState(i.drone_name).kinematics_estimated.position.z_val
            loc = [x,y,z]
            i.drone_locs[self.all_drones.index(i)] = loc
            i.current_location = loc
        
            time.sleep(1)
        self.state_manager = StateManager(self.port)
        self.action_manager = GL_ActionManager(self.port)
        self.client.confirmConnection()
        self.port.get_all_port_statuses()
        self.initial_schedule()
        self.total_delay = 0
        # self.initial_setup()
        #do inital works here

        print('environment initialized')


    def initial_schedule(self): 
        # first = 0
        self.done = False
        for drone in self.all_drones:
            choice = random.randint(0,1) #Controls whether a destination or hover port is picked
            # choice = 0 #All UAMs start by going to destinations
            drone.assign_schedule(port=self.port,client=self.client,choice = choice)
            initial_des = drone.job_status['final_dest']
            if initial_des in self.port.hover_spots:
                drone.set_status("in-action", "in-air")
                drone.port_identification = {'type':'hover','port_no':self.port.hover_spots.index(initial_des)}
            else:
                drone.set_status("in-action", "in-destination")
            drone.job_status['initial_loc'] = drone.current_location
            # print(initial_des)
            self.move_position(drone.drone_name,initial_des,join=0)
        self.update_all()

    def step(self,action):

        self.Try_selecting_drone_from_Start()
        coded_action = self.action_manager.action_decode(self.current_drone, action)
        
        #Ideally, all movements should take place here, and ports that were previously unavailable should free up
        if coded_action["action"] == "land":
            print(["coded action", coded_action, ", current_drone:", self.current_drone.drone_name])
            new_position = coded_action["position"]
            self.complete_landing(self.current_drone.drone_name, new_position)
            #need to calculate reward but 
            old_position = self.current_drone.current_location
            reduce = self.current_drone.calculate_reduction(old_position,new_position) #Ditto
            self.current_drone.set_status("in-action", "in-port")
            # self.current_drone.status = self.current_drone.all_states['in-port']
            self.current_drone.update_battery(reduce) #Ditto
            self.current_drone.job_status["initial_loc"] = self.current_drone.current_location
            self.current_drone.job_status["final_dest"] = new_position

            
        elif coded_action["action"] == "land-b":
            print(["coded action", coded_action, ", current_drone:", self.current_drone.drone_name])
            new_position = coded_action["position"]
            self.complete_landing(self.current_drone.drone_name, new_position)
            #need to calculate reward but 
            old_position = self.current_drone.current_location
            reduce = self.current_drone.calculate_reduction(old_position,new_position) #Ditto
            self.current_drone.set_status("in-action", "battery-port")
            self.current_drone.update_battery(reduce) #Ditto
            self.current_drone.job_status["initial_loc"] = self.current_drone.current_location
            self.current_drone.job_status["final_dest"] = new_position
           # self.uam_time_update(self.current_drone,coded_action['action'])
            
        elif coded_action["action"] == "takeoff":
            
            new_position = coded_action["position"]
            self.complete_takeoff(self.current_drone.drone_name, new_position)
            old_position = self.current_drone.current_location
            reduce = self.current_drone.calculate_reduction(old_position,new_position) #Ditto
            self.current_drone.update_battery(reduce) #Ditto
            self.current_drone.set_status("in-action", "in-destination")
            self.current_drone.job_status["initial_loc"] = self.current_drone.current_location
            self.current_drone.job_status["final_dest"] = new_position
         #   self.uam_time_update(self.current_drone,coded_action['action'])
            #need to calculate reward but how
            

            # self.current_drone.set_status('in-air','in-destination') #Not sure about this one

        elif coded_action["action"] == "hover":
            new_position = coded_action["position"]
            self.move_position(self.current_drone.drone_name, new_position,join=0)
            old_position = self.current_drone.current_location
            reduce = self.current_drone.calculate_reduction(old_position,new_position) #Ditto
            self.current_drone.update_battery(reduce) #Ditto
            self.current_drone.status = self.current_drone.all_states['in-air']
            self.current_drone.job_status["initial_loc"] = self.current_drone.current_location
            self.current_drone.job_status["final_dest"] = new_position
        #    self.uam_time_update(self.current_drone,coded_action['action'])
        
        elif coded_action['action'] == 'takeoff-hover':
            new_position = coded_action["position"]
            self.complete_takeoff_hover(self.current_drone.drone_name,new_position)
            old_position = self.current_drone.current_location
            reduce = self.current_drone.calculate_reduction(old_position,new_position) #Ditto
            self.current_drone.update_battery(reduce) #Ditto
            self.current_drone.status = self.current_drone.all_states['in-air']
            self.current_drone.job_status["initial_loc"] = self.current_drone.current_location
            self.current_drone.job_status["final_dest"] = new_position


        elif coded_action['action'] == 'move-b':
            new_position = coded_action["position"]
            self.change_port(self.current_drone.drone_name, new_position)
            old_position = self.current_drone.current_location            
            reduce = self.current_drone.calculate_reduction(old_position,new_position) #Ditto
            self.current_drone.update_battery(reduce) #Ditto
            self.current_drone.job_status['final_dest'] = new_position
            self.current_drone.status_to_set = self.current_drone.all_states['battery-port']
            self.current_drone.job_status["initial_loc"] = self.current_drone.current_location
            self.current_drone.job_status["final_dest"] = new_position
    #        self.uam_time_update(self.current_drone,coded_action['action'])

        elif coded_action["action"] == "stay":
            self.client.hoverAsync(self.current_drone.drone_name)
        elif coded_action["action"] == "continue":
            pass
        else:
            
            new_position = coded_action["position"]
            self.change_port(self.current_drone.drone_name, new_position)
            old_position = self.current_drone.current_location
            reduce = self.current_drone.calculate_reduction(old_position,new_position) #Ditto
            self.current_drone.update_battery(reduce) #Ditto
            self.current_drone.job_status["initial_loc"] = self.current_drone.current_location
            self.current_drone.job_status["final_dest"] = new_position
            self.current_drone.set_status('in-action','in-port')

        self.update_all()
        if coded_action["action"] != "stay" and coded_action["action"] != "move" and coded_action["action"] != "move-b" and coded_action["action"] != "continue":
            reward = self.calculate_reward_gl(coded_action['action'], coded_action["position"])
        else:
            reward = 0
        self.select_next_drone()
        new_state = self._get_obs()
        self.env_time = (time.time() - self.start_time)  * self.clock_speed                #reduce this if the simulation is too fast
        self.total_timesteps +=1
        # self.debugg()
        done = self.done             #none based on time steps
        if self.total_timesteps >= 100:
            done = True
            self.done = done
        info = {}
        #todo
        #update all the drones

        return new_state,reward,done,info
    
    
    def debugg(self):
        a = [print(i.status) for i in self.all_drones]
        # print(self.client.getMultirotorState("Drone1").kinematics_estimated.position.x_val)
        pass
    
    def complete_takeoff(self,drone_name, fly_port):
        self.client.takeoffAsync(vehicle_name=drone_name).join()
        self.move_position(drone_name, fly_port,join=0)

    def complete_takeoff_hover(self,drone_name, fly_port):
        self.client.takeoffAsync(vehicle_name=drone_name).join()
        self.move_position(drone_name, fly_port,join=1)
        
        
    def move_position(self, drone_name, position,join=0):
        if join == 0:
            self.client.moveToPositionAsync(position[0],position[1],position[2], velocity=1,timeout_sec=15, vehicle_name=drone_name)
            #self.client.hoverAsync(vehicle_name=drone_name)
        else:
            self.client.moveToPositionAsync(position[0],position[1],position[2], velocity=1,timeout_sec=15, vehicle_name=drone_name).join()
            #self.client.hoverAsync( vehicle_name=drone_name)
            
    def complete_landing(self, drone_name, location):        
        self.move_position(drone_name, location,join=1)        
        self.client.landAsync(vehicle_name=drone_name).join()
        
    def select_next_drone(self):                #this one needs to be changed 
        failed_attempts = 0
        if not self.current_drone:
            self.current_drone = self.all_drones[0]
            new_drone_no = 0
        else:
            old_drone_no = self.all_drones.index(self.current_drone)
            new_drone_no = old_drone_no + 1
            
        if new_drone_no < self.no_drones:
            self.current_drone = self.all_drones[new_drone_no]
            drone = self.all_drones[new_drone_no]
            while drone.status == 3 or drone.status == 4:   
                # print(["current drone I am trying to choose1", drone.drone_name])
                #do something to wait for drone 1 to reach destination, so we can start the process again
                time.sleep(2)
                self.update_all()
                failed_attempts+=1
                if failed_attempts > 10: #Failsafe
                    #make done = 1 instead of this
                    self.done = True
                    new_drone_no = old_drone_no+1
                    break

        else:
            self.current_drone = self.all_drones[0]
        self.update_all()
        # print(["status of current drone",self.current_drone.drone_name,'is',self.current_drone.status])

        
    def Try_selecting_drone_from_Start(self):
        """
        When the remianing of the drones are all in air, we can select a drone from the list

        Returns
        -------
        None.

        """
        failed_attempts = 0
        if not self.current_drone: #Means this is the first step
            self.current_drone = self.all_drones[0]
            self.update_all()
            while self.current_drone.status == 3 or self.current_drone.status == 4:   
                time.sleep(1)
                self.update_all()
            return 
        else:
            while self.current_drone.status == 3 or self.current_drone.status == 4:   
                time.sleep(1)
                self.update_all()
                failed_attempts += 1
                if failed_attempts > 30: #Failsafe
                    #make done = 1 instead of this
                    self.done =True 

    
    def enable_control(self, control):
        for i in range(self.no_drones):
            self.client.enableApiControl(False, self.all_drones[i].drone_name)
        for i in range(self.no_drones):
             self.client.armDisarm(False, self.all_drones[i].drone_name)

    def _get_obs(self):
        states = self.state_manager.get_obs(self.current_drone, self.type, self.graph_prop)

        return states




    
    def calculate_reward_gl(self, action, future_loc):
        #pass the decoded action
        #http://lidavidm.github.io/sympy/modules/geometry/line3d.html    - used this
        lambda_ = self.current_drone.schedule_status # Depends on the schedule of the UAM
        beta = self.current_drone.battery_state # Depends on the battery of the UAM
        if self.current_drone.status == self.current_drone.all_states["in-action"]:
            safety = self.calculate_safety(action, future_loc) + self.calculate_safety_2()
        else:
            safety = 1
        print("safety", safety)
        formula = lambda_ * (np.exp(-beta)**2) + safety
        return formula
    
    def calculate_safety_2(self):
        """This is an attempt to determine intersections using 2D points, distance and velocity."""
        threshold = 10  # This number may be too high based on testing. 
        curr_drone = self.current_drone
        curr_pos = curr_drone.current_location
        final_pos  = curr_drone.job_status["final_dest"]
        curr_segment = Segment2D(tuple(curr_pos[:-1]), tuple(final_pos[:-1]))

        other_drones = self.all_drones.copy()
        other_drones.pop(other_drones.index(curr_drone))
        for other_drone in other_drones:
            if other_drone.status == curr_drone.all_states["in-action"]:
                other_loc = other_drone.current_location
                other_final_pos = other_drone.job_status["final_dest"]
                other_segment = Segment2D(tuple(other_loc[:-1]), tuple(other_final_pos[:-1]))
                intersection = curr_segment.intersect(other_segment)
                if intersection:  # Check the distance between each drone and the intersection point, and calc the times.

                    intersection = np.array(intersection.args[0].coordinates).astype(np.float32)
                    curr_segment = np.array(curr_segment.args[0].coordinates).astype(np.float32)
                    other_segment = np.array(other_segment.args[0].coordinates).astype(np.float32)

                    curr_drone_v = self.client.getMultirotorState(vehicle_name=curr_drone.drone_name).kinematics_estimated.linear_velocity
                    other_drone_v = self.client.getMultirotorState(vehicle_name=other_drone.drone_name).kinematics_estimated.linear_velocity

                    inter_norm = np.linalg.norm(intersection)
                    curr_norm, curr_vnorm = np.linalg.norm(curr_segment), np.linalg.norm(np.array([curr_drone_v.x_val, curr_drone_v.y_val]).astype(np.float32))
                    other_norm, other_vnorm = np.linalg.norm(other_segment), np.linalg.norm(np.array([other_drone_v.x_val, other_drone_v.y_val]).astype(np.float32))

                    # Attempting to solve for time of intersection for both drones and checking if the times are too close.
                    # Basically, inter_norm = curr_norm + curr_vnorm * t_intersect, and vice versa for the other drone.

                    ti_curr, ti_other = (inter_norm - curr_norm) / curr_vnorm, (inter_norm - other_norm) / other_vnorm

                    if abs(ti_curr - ti_other) <= threshold:
                        print(f"Drones will collide, intersection 1: {ti_curr}s, intersection 2: {ti_other}s.")
                        return -5
                    else:
                        return 5

        return 0



    def calculate_safety(self, action, future_loc):
        """
        
        calculate 2 parameters for safety
        1. probability on way\
            onway has a circular ring around other drones and if our drones path is in the ring area, we calculate the probability
            if action is takeoff -> P(onway) is calculated with hover spot and destination direction
            if action is landing -> P(onway) is calculated based on current spot and to the landing zone
        2. probability of getting in the route:
             P(get-in route) calculated by 
                 1. drones in hover spot and their possible moving directions(landing spot, takeoff)
                     a. takeoff:
                         i. move to different port
                         ii. takeoff to destination
                     b. landing:
                         i. move to another hover spot
                         ii. land
        Parameters
        ----------
        action : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        #takeoff
        #landing
        #onway
        current_drone_loc = self.current_drone.current_location
        onway_reward = 0
        for i in self.all_drones:
            if self.current_drone is not i:
                drones_locs = i.current_location
                if action == "takeoff":
                    estimated_loc_1 = [current_drone_loc[0] ,current_drone_loc[1] ,current_drone_loc[2] + 10]
                    dist = self.calculate_distance(drones_locs, estimated_loc_1)
                    if dist<1:
                        onway_reward += (1-dist)
                    dist_2 = self.dist_line_point(estimated_loc_1, future_loc, drones_locs)
                    if dist_2 <= 1.5:
                        onway_reward -= 1
                if action == "landing":
                    dist_2 = self.dist_line_point(current_drone_loc, future_loc, drones_locs)
                    if dist_2 <= 1.5:
                        onway_reward -= 1
                        
                        
        in_route_reward = 0
        all_states = self.port.get_all_empty_ports()
        normal_ports = all_states["normal_ports"]
        battery_ports = all_states["battery_ports"]
        hover_ports = all_states["hover_spots"]
        our_drone_segment = Segment3D(tuple(current_drone_loc), tuple(future_loc))
        for i in self.all_drones:
            drone_probs = 1
            if self.current_drone is not i:
                drones_locs = i.current_location
                if i.status == 0:               #in-air = hover spot. Possible actions: land in normal port, land in battery port, move to another hover spot
                    #get free hover spots, get free battery ports, get free ports

                    for port in normal_ports.keys():
                        if normal_ports[port]["occupied"] is False:
                            port_loc = tuple(normal_ports[port]["position"])
                            cur = tuple(drones_locs)
                            a = Segment3D(port_loc, cur)
                            intersect = a.intersection(our_drone_segment)
                            if len(intersect) != 0:
                                drone_probs = drone_probs /2
                    for port in battery_ports.keys():
                        if battery_ports[port]["occupied"] is False:
                            port_loc = tuple(battery_ports[port]["position"])
                            cur = tuple(drones_locs)
                            a = Segment3D(port_loc, cur)
                            intersect = a.intersection(our_drone_segment)
                            if len(intersect) != 0:
                                drone_probs = drone_probs /2                  
                    for port in hover_ports.keys():
                        if hover_ports[port]["occupied"] is False:
                            port_loc = tuple(hover_ports[port]["position"])
                            cur = tuple(drones_locs)
                            a = Segment3D(port_loc, cur)
                            intersect = a.intersection(our_drone_segment)
                            if len(intersect) != 0:
                                drone_probs = drone_probs /2
                elif i.status == 1:           #in-port: possible actions: takeoff(to its destination), move to battery port
                    for port in battery_ports.keys():       #move to battery port requires 3 actions, i just gave one here. needs changing
                        if battery_ports[port]["occupied"] is False:
                            port_loc = tuple(battery_ports[port]["position"])
                            cur = tuple(drones_locs)
                            a = Segment3D(port_loc, cur)
                            intersect = a.intersection(our_drone_segment)
                            if len(intersect) > 0:
                                drone_probs = drone_probs /2
                    end_destination = tuple(i.upcoming_schedule["end-port"])
                    a = Segment3D(drones_locs, end_destination)
                    intersect = a.intersection(our_drone_segment)
                    if len(intersect) > 0:
                        drone_probs = drone_probs /2
                elif i.status == 2:           #in battery port, possible actions: takeoff(to its destination), move to other empty ports
                    end_destination = tuple(i.upcoming_schedule["end-port"])
                    a = Segment3D(drones_locs, end_destination)
                    intersect = a.intersection(our_drone_segment)
                    if len(intersect) > 0:
                        drone_probs = drone_probs/2
                    for port in normal_ports.keys():            #same as before, chaning ports has 3 distinct actions and i only used final destination. needs changing if no good results
                        if normal_ports[port]["occupied"] is False:
                            port_loc = tuple(normal_ports[port]["position"])
                            cur = tuple(drones_locs)
                            a = Segment3D(port_loc, cur)
                            intersect = a.intersection(our_drone_segment)
                            if len(intersect) != 0:
                                drone_probs = drone_probs/2                 
            in_route_reward += drone_probs  
        

        return onway_reward+in_route_reward
            
    
    def update_all(self):
        """
        This function is called during every step, drones should be updated with thecurrent position and status

        Returns
        -------
        None.

        """
        print(["env_time", time.time() - self.start_time])
        for i in self.all_drones:    
            x = self.client.getMultirotorState(i.drone_name).kinematics_estimated.position.x_val
            y = self.client.getMultirotorState(i.drone_name).kinematics_estimated.position.y_val
            z = self.client.getMultirotorState(i.drone_name).kinematics_estimated.position.z_val
            loc = [x,y,z]
            i.current_location = loc
            i.drone_locs[self.all_drones.index(i)] = loc
            self.env_time = (time.time() - self.start_time) * self.clock_speed
            i.update(loc,self.client,self.port,self.env_time)
            i.get_state_status()
            self.drone_feature_mat[self.all_drones.index(i)] = [i.battery_state, i.status, i.schedule_status, x, y] 
        self.port.update_all()
        self.graph_prop['vertiport_features'] = self.port.feature_mat
        self.graph_prop['evtol_features'] = self.drone_feature_mat
        # if self.client.getMultirotorState().collision:
            # print('Oh no.. there was a collision') #procs on ground collisions too... not good... WE can tie this with self.reset if we can get it working correctly
            # self.reset()
        # print("timing")
        # print(self.time_update())
    
    
    def get_locs(self):
        """
        Get the locations of all vehicles to update them in the UAM module

        Returns
        -------
        None.

        """
    
    def reset(self):
        print('resetting')
        self.client.reset()
        time.sleep(5)
        self.enable_control(False)
        self._initialize()
        self.current_drone = self.all_drones[0]
        return self.state_manager.get_obs(self.current_drone, self.type, self.graph_prop)
    
    
    def change_port(self, drone_name, new_port):
        self.client.takeoffAsync(vehicle_name=drone_name).join()
        self.client.moveToPositionAsync(new_port[0],new_port[1],new_port[2], velocity=1, vehicle_name=drone_name).join()
        self.client.landAsync(vehicle_name=drone_name).join()
    
    

    def takeoff(self, drone_name, join=0):
        if join == 0:
            self.client.takeoffAsync(vehicle_name=drone_name)
        else:
            self.client.takeoffAsync(vehicle_name=drone_name).join()
            
    
    
    def get_final_pos(self,port, offset):
        return [port[0] + offset[0] , port[1] + offset[1], port[2]]
    

    def calculate_distance(self, point1,point2):
        
        dist = np.linalg.norm(np.array(point1)-np.array(point2))
        return dist
    
    def dist_line_point(self, p1,p2,p3):
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        dist = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
        return dist

    def create_edge_connect(self,num_nodes=5,adj_mat=None,direct_connect=0):
        """Returns the edge connectivity matrix."""
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
