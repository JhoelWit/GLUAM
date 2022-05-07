from req_cls import UAMs,ports
import airsim
from actionManagerGL import ActionManager
from stateManagerGL import StateManager
from gym import spaces
import gym
import copy

class environment(gym.Env):
    def __init__(self, no_of_drones):
        self.no_drones = no_of_drones
        self.all_drones = list()
        self.current_drone = None
        self.state_manager = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([2,3,3,3,3,2])
        self.drone_offsets = [[0,0], [-6,0], [-4,-4], [-6,-4] ]
        self._initialise()
    
    def _initialise(self):
        self.port = ports(self.no_drones)
        self.client = airsim.MultirotorClient()
        self.enable_control()
        for i in range(self.no_drones):
            drone_name = "Drone"+str(i)
            offset = self.drone_offsets[i]
            self.all_drones.append(UAMs(drone_name, offset))
            self.client.enableApiControl(True, drone_name)
            self.client.armDisarm(True, drone_name)
        self.port = ports(self.no_drones)
        self.state_manager = StateManager(self.port)
        self.action_manager = ActionManager(self.port)
        self.current_drone = self.all_drones[0]
        
    def step(self,action):
        coded_action = self.action_manager.action_decode(self.current_drone, action)
        print('coded action\n',coded_action)
        if coded_action["action"] == "land":
            new_position = coded_action["position"]
            self.complete_landing(self.current_drone, new_position)
            #need to calculate reward but how
        elif coded_action["action"] == "takeoff":
            new_position = coded_action["position"]
            self.complete_takeoff(self.drone_name, new_position)
            #need to calculate reward but how
        else:
            pass
        reward = self.calculate_reward()
        self.select_next_drone()
        new_state = self._get_obs()
        done = None                 #none based on time steps
        info = {}
        return new_state,reward,done,info
    
    def complete_takeoff(self,drone_name, fly_port):
        self.client.takeoffAsync(vehicle_name=drone_name).join()
        self.move_position(drone_name, fly_port )
        
        
    def move_position(self, drone_name, position):
        print('drone name',drone_name,'position',position)
        self.client.moveToPositionAsync(position[0],position[1],position[2], velocity=1, vehicle_name=drone_name)
    
    def complete_landing(self, drone_name, location):
        self.move_position(drone_name, location )        
        self.client.landAsync(vehicle_name=drone_name)
        
    def select_next_drone(self):
        print('current drone',self.current_drone)
        cur_drone = copy.copy(self.current_drone) #Use copy.copy or copy.deepcopy depending on what you want to transfer, neither work for indexing on the next line though.
        current_drone_no = self.all_drones.index(self.current_drone)
        print('current drone no',current_drone_no)
        # current_drone_no = int(cur_drone[1])
        new_drone = current_drone_no + 1
        self.current_drone = self.all_drones[new_drone]
    
    def _get_obs(self):
        states = self.state_manager.get_obs(self.current_drone)
        return states
    
    def calculate_reward(self):
        # B = i/10
        # min_b2 = -B**2
        # R = -(i/N_of_uams) + (j * math.exp(min_b2/10))
        return 0
    
    def reset(self):
        pass