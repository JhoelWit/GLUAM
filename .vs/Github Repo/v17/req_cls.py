# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 12:51:19 2022

@author: praji
"""
import random
import math 



class ports:
    def __init__(self,drone_count):
        self.normal_ports = [[0,0], [-3,0]]
        self.fake_ports = [[13,4,-10], [10,1,-10], [-10,1,-10]]
        self.hover_spots = [[13,4,-2], [10,1,-2], [-10,1,-2]]
        self.battery_ports = [[-2,3,-2]]
        self.no_ports = len(self.normal_ports)
        self.no_battery_ports = len(self.battery_ports)
        self.no_hoverspots = len(self.hover_spots)
        self.port_status = {}
        self.port_center_loc =[]
        self.drone_count = drone_count
        self.dist_threshold = 10
        for i in range(self.no_ports):
            self.port_status[i] = {"port_no": i, "position":self.normal_ports[i],"occupied": False}
            
        self.battery_port_status = {}
        for i in range(self.no_battery_ports):
            self.battery_port_status[i] = {"port_no": i,"position":self.battery_ports[i],"occupied": False}
            
        self.hover_spot_status = {}
        for i in range(self.no_hoverspots):
            self.hover_spot_status[i] = {"port_no": i,"position":self.hover_spots[i],"occupied": False}
            
            
    def get_empty_port(self):
        for i in range(self.no_ports):
            if self.port_status[i]["occupied"] == False:
                return self.port_status[i]
    
    def get_empty_battery_port(self):
        for i in range(self.no_battery_ports):
            if self.port_status[i]["occupied"] == False:
                return self.battery_port_status[i]
    
    def get_empty_hover_Spot(self):
        for i in range(self.no_hoverspots):
            if self.port_status[i]["occupied"] == False:
                return self.hover_spot_status[i]
            
    def change_status_normal_port(self, port_no, occupied):
        self.port_status[port_no]["occupied"] = occupied
        
    def change_status_battery_port(self, port_no, occupied):
        self.battery_port_status[port_no]["occupied"] = occupied
    
    def change_hover_spot_status(self, port_no, occupied):
        self.hover_spot_status[port_no]["occupied"] = occupied
            
    def get_count_empty_port(self):
        cnt = 0
        for i in range(self.no_ports):
            if self.port_status[i]["occupied"] == False:
                cnt+=1
        return cnt
    
    def get_count_empty_battery_port(self):
        cnt = 0
        for i in range(self.no_battery_ports):
            if self.port_status[i]["occupied"] == False:
                cnt+=1
        return cnt
    
    def get_count_empty_hover_Spot(self):
        cnt = 0
        for i in range(self.no_hoverspots):
            if self.port_status[i]["occupied"] == False:
                cnt+=1
        return cnt   
    
    def get_availability_ports(self,drone_locs):
        empty_ports = self.get_count_empty_port()
        uams_inside = self.count_uavs_inside(drone_locs)
        percent = empty_ports/uams_inside
        if percent > 0.8:
            return 2
        elif percent> 0.5:
            return 1
        else:
            return 0
        
    
    def get_availability_battery_ports(self,drone_locs):
        empty_ports = self.get_count_empty_battery_port()
        uams_inside = self.count_uavs_inside(drone_locs)
        percent = empty_ports/uams_inside
        if percent > 0.8:
            return 2
        elif percent> 0.5:
            return 1
        else:
            return 0
        
    def get_availability_hover_spots(self,drone_locs):
        empty_ports = self.get_count_empty_hover_Spot()
        uams_inside = self.count_uavs_inside(drone_locs)
        percent = empty_ports/uams_inside
        if percent > 0.8:
            return 2
        elif percent> 0.5:
            return 1
        else:
            return 0
    
    def port_status(self):
        pass
    
    def get_destination(self):
        return random.choice(self.fake_ports)
        #self.fake_ports
    
    def count_uavs_inside(self,drone_locs):
        UAVs_inside = 0
        for i in range(len(drone_locs)):
            dist= self._calculate_distance(drone_locs[i])
            if dist>self.dist_threshold:
                UAVs_inside +=1
        return UAVs_inside
    
    def _calculate_distance(self,cur_location):
        return math.dist(self.port_center_loc, cur_location)
    
    


class UAMs:
    def __init__(self, drone_name,offset):
        self.drone_name = drone_name
        self.drone_no = drone_name # use split and get the drone number alone
        self.battery_remaining = 100
        self.distance_travelled = 0
        self.next_takeoff = None
        self.next_landing = None
        self.all_states = {"in-air":0, "in-port":1, "battery_port":2}
        self.status = 0
        self.offset = offset
        self.current_location = []
        self.in_portzone = False
        self.port_center_loc =[]
        self.dist_threshold = 10
    
    def get_status(self):
        return self.status
    
    def set_status(self,status):
        self.status = self.all_states[status]
    
    def get_schedule_state(self):
        pass
    
    def set_schedule(self,whatever):
        pass
    
    def get_battery_state(self):
        return self.battery_remaining
    
    def _update_battery(self, reduce):
        self.battery_remaining -= reduce
        
    def distance_to_nearest_drone(self, drone_no):
        #we can use it later
        pass
    
    def check_zone(self):
        dist = self._calculate_distance(self.current_location)
        if dist<self.dist_threshold:
            self.in_portzone = True
        else:
            self.in_portzone = False
    
    def _calculate_distance(self,cur_location):
        return math.dist(self.port_center_loc, cur_location)
    
    def land(self, client):
        pass
    
    
    
