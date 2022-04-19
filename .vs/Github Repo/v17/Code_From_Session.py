import numpy as np
import operator
import matplotlib.pyplot as plt
import time
%matplotlib inline
global time
weatherStates = ['WIND GUST','WIND BREEZE','HIGH OBSTACLE','LOW OBSTACLE','NORMAL']
class ATC:
    # trajecory --> call some path planning algo-->[1,2,3,3,4,6]
    # schedule ---> function? member variable? time?
    # schedule() ---> [ {'starting_pad' :, 'ending_pad', 'start_time', 'end_time' }, , ]
    # weather ---> 
    def __init__(self, vehicles = 5, pads = 3):
        self.vehicles = vehicles
        self.pads = pads
        self.vehicles = []
    
    
    def compute(self):
        reward = 0
        for vehicle in self.vehicles:
            n = np.random.randint(0,len(self.weatherStates))
            weather = self.weatherStates(n)
            # add in the uncertainities to the position
            vehicle.update_state(weather)
            state = vehicle.state
            action = get_action(state)
            reward += get_reward(action)
        return reward
            
class Vehicle:
    def __init__(self, drone, start):
        # airsim object
        self.drone = drone 
        self.state = {'airsim_state' : None , 'uncertain_state' : start ,  'battery' : 1 }
    
    def update(self, weather, vehicle_name):
        self.state['airsim'] = self.drone.GetMultiRotorState(vehicle_name = vehicle_name)
        # self.state['uncertain_state'][0] += 


        







# class Vertiport(num_ports=3):
#     # Initialize the environment variables
#     def __init__(self,num_ports):
#         # Environment details
#         self.height = 5 * num_ports
#         self.width = 5 * num_ports
#         self.grid = np.zeros((  self.height , self.width  )) # initial reward will be 0 for each state

#         # Starting position for the agent
#         self.startpts = [] # To keep track of the different agent starting points and make sure they don't conflict
#         self.current_location = (np.floor(self.height / 2) , np.random.randint(0,self.width))
#         self.start_location = self.current_location
#         self.previous_location = self.current_location
#         self.startpts.append(self.start_location)

#         # Set locations for each vertiport
#         self.port_locations = [ [] for ports in range(num_ports) ]
#         self.terminal_states = []
#         for i in range(num_ports):
#             location = (np.floor( self.height / i ), np.random.randint(0,self.width))
#             self.port_locations[i] = location
#             self.terminal_states.append(location)
        
#         # Set rewards for each boundary state (ports)
#         self.port_reward = 10
#         for i in range(num_ports):
#             self.grid[self.port_locations[i][0],self.port_locations[i][1]] = self.port_reward 

#         # Action space
#         self.actions = ['HIGH', 'LOW', 'SPEED UP', 'SLOW DOWN','LAND','TAKEOFF','RETURN']




# class ATC(num_agents = 5,velocity = 1):
#     def __init__(self, gamma=1):
#         self.weatherStates = ['WIND GUST','WIND BREEZE','HIGH OBSTACLE','LOW OBSTACLE','NORMAL']
#         self.q_value = [0 for agents in range(num_agents)]
#         self.gamma = gamma