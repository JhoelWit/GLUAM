# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:39:16 2022

@author: Prajit
"""
import numpy as np

class StateManager:
    def __init__(self, ports):
        self.ports = ports
        
    
    def get_obs(self, drone, type, graph_prop = None):
        """
        
        
        Battery capacity of current vehicle(0,1)
        Empty ports (0,1,2)
            3.1 Not available – 0
            3.2 availability limited – 1
            3.3 Sufficient ports available - 2
        empty hovering spots(0,1,2)
        (same choices as empty ports)
        Battery ports(0,1,2)
        (same choices as empty ports)
        Status of the vehicle(0,1,2)
          6.1 in-air– 0
          6.2 in-port– 1
          6.3 in-battery port- 2
        On-time to takeoff/land (0,1)
       # not implemented -  Collision possibility (0,1,2) => 0 – safe; 2 – highly unsafe


        Returns
        -------
        None.

        """
        def generate_mask():
            """Returns a mask to use for the GRL and RL policy models."""
            if drone.status == drone.all_states["in-action"]:
                mask = [1] * 13
                mask[-1] = 0
                return np.array(mask)

            mask = [0] * 13
            for row_idx, row in self.ports.feature_matrix:  # Looks at availability of each port and masks if the port is not available
                mask[row_idx + 6] = 1 if row[0] == 0 else 0
            
            return np.array(mask)


        print(drone.get_all_status())
        drone_locs = drone.drone_locs 
        battery_capacity = drone.get_battery_state()
        empty_port = self.ports.get_availability_ports(drone_locs)                       
        empty_hovering_spots = self.ports.get_availability_hover_spots(drone_locs)        
        empty_battery_ports = self.ports.get_availability_battery_ports(drone_locs)       
        status = drone.get_status()
        schedule = drone.get_state_status() 
        states = np.array([battery_capacity,
                            empty_port,
                            empty_hovering_spots,
                            empty_battery_ports,
                            status,
                            schedule, 
                            drone.current_location[0], 
                            drone.current_location[1], 
                            drone.current_location[2]])
        if type == "regular":
            return states
            
        elif type == "graph":
            graph_prop['next_drone_embedding'] = states
            graph_prop["mask"] = generate_mask()
            return graph_prop
    
    
    
    def drones_search(self):
        pass
    
    
class GL_StateManager:
    def __init__(self, ports, drones):
        self.ports = ports
        self.drones = drones
        
    def get_obs(self, current_drone):
        ports = self.ports.get_all_port_statuses()
        for i in self.drones:
            pass
       # print(self.ports.get_all_port_statuses())