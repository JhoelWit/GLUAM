# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 12:51:19 2022

@author: ADAMS-LAB
"""
import random
import math 
import numpy as np

class ports:
    """
    A class which has details about all the port locations.
    """
    def __init__(self,drone_count):
        self.normal_ports = [[0,0,-2], [-3,0,-2]]
        # self.normal_ports = [[51.6,11.2,-1], [51.6,14.3,-2], [0.1,0,-1], [-2,3,-1], [-3,0,-3], [48.5,11.10,-4], [48.50,14.40,-2], [45.40,11.30,-1], [45.40,14.40,0], [15.90,63.20,-7], [12.80,63.20,-9], [9.8,63.20,-10], [6.7,63,-6]] #For the demo
        self.fake_ports = [[0,3,0], [-11,1,0], [-8,6,0], [-8,-5,0], [-12,8,0]] #These are closer
        # self.fake_ports = [[0,3,0], [-11,1,0], [-8,6,0], [-8,-5,0], [-12,8,0],[42.30,9,0],[42.30,17,0],[47,9,0],[47,17,0],[20,60,0],[6,60,0],[-3,-7,0]] #For the demo
        self.hover_spots = [[-1,-9,0], [-9,-9,0],[-10,0,0], [-5,12,0], [-8,12,0], [2,4,0],[-5,-9,0]]
        # self.hover_spots = [[-1,-9,0], [-9,-9,0],[-10,0,0], [-5,12,0], [-8,12,0], [2,4,0],[-5,-9,0], [13.7,0,0], [15,0,0], [17,0,0], [20,0,0], [-4,0,0], [-4,3,0], [-4,6,0], [-7,0,0], [-7,3,0], [-7,6,0]] #For the demo
        self.battery_ports = [[-2,3,-2]]
        # self.battery_ports = [[-2,3,-2], [42.30,11.20,-3], [42.30,14.50,-4], [9.8,60.10,-9], [15.90,60.10,-8]] #For the demo
        self.no_ports = len(self.normal_ports)
        self.no_battery_ports = len(self.battery_ports)
        self.no_hoverspots = len(self.hover_spots)
        self.no_total = self.no_ports + self.no_battery_ports + self.no_hoverspots
        self.feature_mat = np.zeros((self.no_total,2)) #two features per port
        self.port_status = {}
        self.port_center_loc =[0,0,-4] #Filler
        self.drone_count = drone_count
        self.dist_threshold = 10
        self.reset_ports()
    
    def reset_ports(self):
        for i in range(self.no_ports):
            self.port_status[i] = {"port_no": i, "position":self.normal_ports[i],"occupied": False}
        self.battery_port_status = {}
        for i in range(self.no_battery_ports):
            self.battery_port_status[i] = {"port_no": i,"position":self.battery_ports[i],"occupied": False}
        self.hover_spot_status = {}
        for i in range(self.no_hoverspots):
            self.hover_spot_status[i] = {"port_no": i,"position":self.hover_spots[i],"occupied": False}


    def update_port(self,port):
        if port:
            if port['type'] == 'normal':
                # print('\nport relinquished\n')
                self.change_status_normal_port(port['port_no'],False)
            elif port['type'] == 'battery':
                # print('\nbattery port relinquished\n')
                self.change_status_battery_port(port['port_no'],False)
            elif port['type'] == 'hover':
                # print('\nhover spot relinquished\n')
                self.change_hover_spot_status(port['port_no'],False)

    def update_all(self):
        '''''
        This function will iterate through all ports, battery ports, and hover spots and 
        update the vertiport feature matrix accordingly
        '''''
        for i in range(self.no_ports):
            if self.port_status[i]['occupied'] == True:
                availability = 0
            else:
                availability = 1
            node_type = 0
            self.feature_mat[i] = [availability,node_type]
        for i in range(self.no_battery_ports):
            if self.battery_port_status[i]['occupied'] == True:
                availability = 0
            else:
                availability = 1
            node_type = 1
            self.feature_mat[i+self.no_ports] = [availability,node_type]
        for i in range(self.no_hoverspots):
            if self.hover_spot_status[i]['occupied'] == True:
                availability = 0
            else:
                availability = 1
            node_type = 2
            self.feature_mat[i+self.no_ports+self.no_battery_ports] = [availability,node_type]
            
    def get_empty_port(self):
        for i in range(self.no_ports):
            if self.port_status[i]["occupied"] == False:
                self.change_status_normal_port(self.port_status[i]['port_no'],True)
                return self.port_status[i]
    
    def get_empty_battery_port(self):
        for i in range(self.no_battery_ports):
            if self.battery_port_status[i]["occupied"] == False:
                self.change_status_battery_port(self.battery_port_status[i]['port_no'],True)
                return self.battery_port_status[i]
        return None
    
    def get_empty_hover_status(self):
        for i in range(self.no_hoverspots):
            if self.hover_spot_status[i]["occupied"] == False:
                self.change_hover_spot_status(self.hover_spot_status[i]['port_no'],True)
                return self.hover_spot_status[i]

    def get_destination(self,choice = 0):
        if choice == 0:
            return random.choice(self.fake_ports)
        else:
            empty_port = self.get_empty_hover_status()
            return empty_port['position']

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
            if self.hover_spot_status[i]["occupied"] == False:
                cnt+=1

        return cnt   
    
    def get_availability_ports(self,drone_locs):
        empty_ports = self.get_count_empty_port()
        percent = empty_ports / self.no_ports
        # uams_inside = self.count_uavs_inside(drone_locs)
        if percent >= 0.8:
            return 2
        elif percent >= 0.5:
            return 1
        else:
            return 0

    def get_availability_battery_ports(self,drone_locs):
        empty_ports = self.get_count_empty_battery_port()
        # uams_inside = self.count_uavs_inside(drone_locs)
        percent = empty_ports / self.no_battery_ports
        if percent >= 0.8:
            return 2
        elif percent >= 0.5:
            return 1
        else:
            return 0
       
    def get_availability_hover_spots(self,drone_locs):
        empty_ports = self.get_count_empty_hover_Spot()
        # uams_inside = self.count_uavs_inside(drone_locs)
        percent = empty_ports / self.no_hoverspots
        if percent >= 0.8:
            return 2
        elif percent >= 0.5:
            return 1
        else:
            return 0
    
    def count_uavs_inside(self,drone_locs):
        UAVs_inside = 0
        for i in range(len(drone_locs)):
            dist= self._calculate_distance(drone_locs[i])
            if dist<self.dist_threshold: #Switched from > to <
                UAVs_inside +=1
        return UAVs_inside
    
    def _calculate_distance(self,cur_location):
        return np.linalg.norm(np.array(self.port_center_loc)-np.array(cur_location)) #math.dist starts at python3.8, I'm using 3.7 lol
    
class UAMs:
    def __init__(self, drone_name,offset):
        self.drone_name = drone_name
        self.drone_no = int(drone_name[-1]) 
        self.velocity = 1 # 1 m/s
        self.all_battery_states = {'critical':0,'sufficient':1,'full':2} #Added 4.25.22 -> 3 different battery states to go with the overall battery remaining
        self.battery_state = 2
        self.flight_time = 100
        self.distance_travelled = 0
        self.next_takeoff = None
        self.next_landing = None
        self.all_states = {"in-air":0, "in-port":1, "battery-port":2, "in-action":3, "in-destination":4}
        self.job_status = {"initial_loc":None, "final_dest":None, "current_pos": None}
        self.status = 1
        self.status_to_set = 4
        self.offset = offset
        self.current_location = []
        self.in_portzone = False
        self.port_center_loc =[0,0,-4] #Filler
        self.dist_threshold = 10 #Distance threshold for collision avoidance
        self.drone_locs = [[0,0,-1],[6,0,-1],[-2,3,-1],[6,4,-1],[-3,0,-1]]
        # self.drone_locs = [[-51.6,-11.2,-1], [-51.6,-14.3,-2], [2,-3,-1], [-6,-4,-2], [3,0,-3], [-48.5,-11.10,-4], [-48.50,-14.40,-2], [-45.40,-11.30,-1], [-45.40,-14.40,0], [-42.30,-11.20,-3], [-42.30,-14.50,-4], [-15.90,-63.20,-7], [-15.90,-60.10,-8], [-12.80,-63.20,-9], [-9.8,-63.20,-10], [-9.8,-60.10,-9], [-6.7,-63,-6]] #For the demo
        self.current_location = None
        self.in_battery_port = 0
        self.collision_status = 0 #Can be 0 or 1
        self.schedule_status = 0 #Can be 0 or 1
        self.tasks_completed = 0
        self.port_identification = None
        self.upcoming_schedule = {"landing-time": None, "takeoff-time":None, 'landing-delay': None,'takeoff-delay':None,'time':0}

    def get_status(self):
        if self.status == self.all_states['in-air']:
            status = 0
        else:
            status = 1
        return status
    
    def set_status(self,status, final_status):
        self.status = self.all_states[status]
        self.status_to_set = self.all_states[final_status]
    
    def get_schedule_state(self):
        if (self.upcoming_schedule['landing-time'] - 1) <= self.upcoming_schedule['time'] <= (self.upcoming_schedule['landing-time'] + 1):
            schedule = 0
        else:
            schedule = 1
        if (self.upcoming_schedule['takeoff-time'] - 1) <= self.upcoming_schedule['time'] <= (self.upcoming_schedule['takeoff-time'] + 1):
            schedule = 0
        else:
            schedule = 1
        return schedule 
    
    def calculate_reduction(self,old_position,new_position): 
        time_travelled = np.linalg.norm(np.array(old_position)-np.array(new_position)) / self.velocity
        reduction = time_travelled / 60 #Conversion to minutes
        if reduction < 1:
            return 1
        else:
            return reduction

    def update_flight_time(self, reduce):
        self.flight_time -= reduce
        if self.flight_time < 0:
            self.flight_time = 0 
        if self.flight_time == 30:
            self.battery_state = self.all_battery_states['full']
        elif 9 <= self.flight_time <= 30: #Added 4.25.22 
            self.battery_state = self.all_battery_states['sufficient'] #Ditto
        elif self.flight_time < 9:
            self.battery_state = self.all_battery_states['critical'] #Ditto
        
        
    def collision_avoidance(self):
        drone_no = self.drone_no
        locs_to_check = self.drone_locs
        drone_loc = self.drone_locs[drone_no]
        for drone_loc2 in locs_to_check:
            dist = self._calculate_distance(drone_loc,drone_loc2)
            zdist = self._calculate_distance(drone_loc[-1],drone_loc2[-1])
            if (dist < self.dist_threshold) and (drone_loc != drone_loc2):
                if (zdist < 3): #If the drones are too close to each other, which means they could possibly collide
                    # print('drones are close')
                    self.collision_status = 1
                    return
                else:
                    self.collision_status = 0
        
    def check_zone(self):
        dist = self._calculate_distance(self.current_location)
        if dist<self.dist_threshold:
            self.in_portzone = True
        else:
            self.in_portzone = False
    
    def update(self, current_loc, client,port):
        self.current_location = current_loc
        if self.status == self.all_states['in-action']: 
            if self.status_to_set == self.all_states['in-destination']: 
                dist = self._calculate_distance(current_loc[:-1],self.job_status['final_dest'][:-1]) #Adding [:-1] since the z height isn't as important and will be constantly changing, it will have to land after
                if dist < 5: #Drone reached destination and is ready for the next task, changed to 5 to avoid duplication issue
                    # print(["status of ", self.drone_name ," changed from ", self.status ," to ", self.status_to_set])
                    self.set_status('in-destination','in-action')
                    self.tasks_completed += 1
                else:
                    final_pos = self.job_status['final_dest']
                    client.moveToPositionAsync(final_pos[0],final_pos[1],self.offset[-1], velocity=1, vehicle_name=self.drone_name)
                    # print(["status of ", self.drone_name ," should change from ", self.status ," to ", self.status_to_set, "current dist needed is", dist])

            elif self.status_to_set == self.all_states['battery-port']:
                dist = self._calculate_distance(current_loc[:-1],self.job_status['final_dest'][:-1])
                if dist < 1: #Drone reached the battery port and is ready to charge
                    client.landAsync(vehicle_name = self.drone_name).join()
                    self.in_battery_port = 1
                    # print(["status of ", self.drone_name ," changed from ", self.status ," to ", self.status_to_set])
                    self.set_status('battery-port','in-action')
                    self.flight_time += 6
                    # print('battery remaining',self.flight_time)
                else:
                    final_pos = self.job_status['final_dest']
                    client.moveToPositionAsync(final_pos[0],final_pos[1],self.offset[-1], velocity=1, vehicle_name=self.drone_name)
                    # print(["status of ", self.drone_name ," should change from ", self.status ," to ", self.status_to_set, "current dist needed is", dist])

            elif self.status_to_set == self.all_states['in-air']:
                dist = self._calculate_distance(current_loc[:-1],self.job_status['final_dest'][:-1])
                if dist < 1: #Drone reached the hover spot and is ready for the next task
                    client.hoverAsync(vehicle_name = self.drone_name)
                    old_position = current_loc
                    new_position = self.job_status['final_dest']
                    reduce = self.calculate_reduction(old_position,new_position) #Ditto
                    self.update_flight_time(reduce) #Ditto
                    # print(["status of ", self.drone_name ," changed from ", self.status ," to ", self.status_to_set])
                    self.set_status('in-air','in-action')
                else:
                    final_pos = self.job_status['final_dest']
                    client.moveToPositionAsync(final_pos[0],final_pos[1],self.offset[-1], velocity=1, vehicle_name=self.drone_name)
                    # print(["status of ", self.drone_name ," should change from ", self.status ," to ", self.status_to_set, "current dist needed is", dist])
            elif self.status_to_set == self.all_states['in-port']:
                dist = self._calculate_distance(current_loc[:-1],self.job_status['final_dest'][:-1])
                if dist < 1: #Drone reached destination and is ready for the next task
                    # print(["status of ", self.drone_name ," changed from ", self.status ," to ", self.status_to_set])
                    self.set_status('in-port','in-action')
                else:
                    final_pos = self.job_status['final_dest']
                    client.moveToPositionAsync(final_pos[0],final_pos[1],self.offset[-1], velocity=1, vehicle_name=self.drone_name)
                    # print(["status of ", self.drone_name ," should change from ", self.status ," to ", self.status_to_set, "current dist needed is", dist])

        elif self.status == self.all_states['battery-port']:
            # print('drone',self.drone_name,' is charging')
            if self.flight_time >= 30:
                self.flight_time = 30
                #self.assign_schedule(port,choice = 0)
                self.set_status('battery-port','in-action')
            else:
                self.flight_time += 6
                # print('battery remaining1',self.flight_time)

        elif self.status == self.all_states['in-port']:
           # self.assign_schedule(port,choice=0)
            # client.takeoffAsync(vehicle_name=self.drone_name)
            self.set_status('in-port','in-action')
            # print('[ \n status of ', self.drone_name ,"should change from ", self.status ," to ", self.status_to_set,']')

        elif self.status == self.all_states['in-air']:
            # print('drone',self.drone_name,'is currently hovering! It''s ready for an action')
            pass

        elif self.status == self.all_states['in-destination']:
            self.assign_schedule(port,choice=1) #Assigning a hover port
            self.port_identification = {'type':'hover','port_no':port.hover_spots.index(self.job_status['final_dest'])}
            des = self.job_status['final_dest']
            final_pos = self.get_final_pos(des, self.offset)
            client.moveToPositionAsync(final_pos[0],final_pos[1],self.offset[-1], velocity=1, vehicle_name=self.drone_name)
            self.set_status('in-action','in-air')
            # print('drone',self.drone_name,'is moving')
            # print(["status of ", self.drone_name ,"should changed from ", self.status ," to ", self.status_to_set])
        
    def _calculate_distance(self,cur_location, dest):
        return np.linalg.norm(np.array(dest)-np.array(cur_location))
    
    def update_port(self, is_it):
        self.in_battery_port = is_it
        
    def get_final_pos(self,port, offset):
        return [port[0] + offset[0] , port[1] + offset[1], offset[2]]
    
    def assign_schedule(self,port,choice = 0):
        """
        Instead of the schedule class, everytime the drone reaches its destination(fake ports) I assign another schedule. 

        Returns
        -------
        None.

        """
        self.job_status['final_dest'] = port.get_destination(choice)
        random_landing = random.randint(1,2) * 10
        random_takeoff = random.randint(1,2) * 10
        self.upcoming_schedule["landing-time"] = random_landing + self.upcoming_schedule['time']
        self.upcoming_schedule["takeoff-time"] = random_takeoff + self.upcoming_schedule['time']

    def update_time(self,loc,update):
        time = (self._calculate_distance(loc,self.current_location) / self.velocity) / 60 #Want the time in minutes from now on
        if time < update: 
            #drone is hovering or otherwise staying still
            time = update
        self.upcoming_schedule['time'] += time
        # print(time)
        # print('new time for drone:',self.drone_no,self.upcoming_schedule['time'])
        # print('landing time for drone:',self.drone_no,self.upcoming_schedule['landing-time'])
        # print('takeoff time for drone:',self.drone_no,self.upcoming_schedule['takeoff-time'])

    def get_state_status(self):
        """
        Our state space(On-time to takeoff/land (0,1)) indicates the takeoff and landing time, delay. Please calculate them here
        For perfect takeoff - you can have threshold of 1 minute. Create new variable, check its timing if it is good timing set it to 1 else 0
        if there is delay just calculate them for the reward claculation
        1. For landing the delay time is from the time mentioned in the self.upcoming_schedule["Landing-time"]
        2. for takeoff the delay time is from the time mentioned in the self.upcoming_schedule["takeoff-time"]

        Returns
        -------
        None.

        """
        if self.status == self.all_states['in-air'] or self.status == self.all_states['in-action']:
            if (self.upcoming_schedule["landing-time"] - 5 <= self.upcoming_schedule['time'] <= self.upcoming_schedule["landing-time"] + 5):
                self.schedule_status = 0
                return 0
            elif ( self.upcoming_schedule['time'] <= self.upcoming_schedule["landing-time"] - 5):
                self.schedule_status = 1
                return 1
            elif (self.upcoming_schedule["landing-time"] - 5 <= self.upcoming_schedule['time']):
                self.schedule_status = 2
                return 2
        elif self.status == self.all_states['in-port'] or self.status == self.all_states['battery-port']:
            if (self.upcoming_schedule["takeoff-time"] - 5 <= self.upcoming_schedule['time'] <= self.upcoming_schedule["takeoff-time"] + 5):
                self.schedule_status = 0
                return 0
            elif (self.upcoming_schedule['time'] <= self.upcoming_schedule["takeoff-time"] - 5):
                self.schedule_status = 1
                return 1
            elif (self.upcoming_schedule["takeoff-time"] + 5 <= self.upcoming_schedule['time']):
                self.schedule_status = 2
                return 2
        else:
            self.schedule_status = 0
            return 0
        
    


