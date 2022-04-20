# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import string


class UAM():
    def __init__(self, uam_id:string,uam_config):
        #uam initialization parameters
        self.uam_id = uam_id
        self.battery = uam_config[uam_id]['battery']
        self.current_mode = uam_config[uam_id]['mode']
        self.pos = uam_config[uam_id]['position']
        pass
    def get_status(self):
        pass
    def get_battery_state(self):
        pass
    def _update_battery(self,drone_no):
        pass
    def distance_to_nearest_drone(self, drone_no):
        pass
    def _battery_required(self, drone_no, start_pos,goal_pos):
        pass


class PORT():
    def __init__(self, port_id:string, port_config,schedule):
        #port initialization parameters
        self.port_id = port_id
        self.time_int = port_config[port_id]['time_int']
        self.next_uam = port_config[port_id]['uam']
        self.available = port_config[port_id]['availability']
        self.port_type = port_config[port_id]['port_type']
        self.schedule = schedule[port_id]
        pass
    def get_status(self):
        pass
    def get_schedule_state(self):
        pass


uam_config = {
'uam_0' : {
'battery' : 1,
'mode' : 'battery port',
'position' : (1,1,0)
},
'uam_1' : {
'battery' : 0.7,
'mode' : 'hovering',
'position' : (3,3,5)
},
'uam_2' : {
'battery' : 0.5,
'mode' : 'on path',
'position' : (27,3,5)
},
'uam_3' : {
'battery' : 0.9,
'mode' : 'landing pad',
'position' : (1,3,0)
},
'uam_4' : {
'battery' : 0.3,
'mode' : 'battery port',
'position' : (0,0,0)
},
}

port_config = {
'port_0' : {
'time_int' : 35,
'uam' : 'uam_0',
'availability' : 'no',
'port_type' : 'landing'
},
'port_1' : {
'time_int' : 5,
'uam' : 'uam_1',
'availability' : 'yes',
'port_type' : 'hovering'
},
'port_2' : {
'time_int' : 20,
'uam' : 'uam_2',
'availability' : 'no',
'port_type' : 'battery'
},
}

schedule = {
'port_0' : {
'uam_0' : 35,
'uam_3' : 40
},
'port_1' : {
'uam_1' : 5,
'uam_4' : 10
},
'port_2' : {
'uam_2' : 20,
},
}