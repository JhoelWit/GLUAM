# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:39:16 2022

@author: Prajit
"""

class StateManager:
    def __init__(self, ports):
        self.ports = ports
        
    
    def get_obs(self, drone):
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
        states = list()
        battery_capacity = drone.get_battery_state()
        empty_port = self.ports.get_availability_ports()                        #need another way to get all drone locs
        empty_hovering_spots = self.ports.get_availability_hover_spots()        #need another way to get all drone locs
        empty_battery_ports = self.ports.get_availability_battery_ports()       #need another way###
        status = drone.get_status()
        schedule = drone.get_schedule_state()
        states = [battery_capacity,empty_port,empty_hovering_spots,empty_battery_ports,status, schedule ]
        return states
    
    
    
    def drones_search(self):
        pass