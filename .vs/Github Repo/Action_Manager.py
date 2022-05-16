# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:38:39 2022

@author: ADAMS-LAB
"""

class ActionManager:
    """
    This class should decode the actions and send the cooridnate to the Main environment
    """
    def __init__(self, port):
        self.port = port

    
    def action_decode(self,drone, action):
        """
        First check the status then assign the port/hoveringspot/takeoff/land .... etc

        Parameters
        ----------
        drone : drone object
            DESCRIPTION.
        action : just a scalar value [0-4]
            DESCRIPTION.

        Returns
        -------
        None.

        """
        status = drone.status
        if status == drone.all_states['in-air']:
            #here we are in landing phase, return 0 - 4 stay still, land, battery port, empty hovering spot
            if action ==0:
                return {"action": "stay", "position" : None}
            elif action ==1: #land
                empty_port = self.port.get_empty_port()
                if not empty_port:
                    return {"position" : None, "action": "stay-penalty"} #Trying to add a penalty here instead of stay, to give termination criteria to the model
                else:
                    self.port.change_status_normal_port(empty_port["port_no"], True)
                    final_pos = self.get_final_pos(empty_port["position"], drone.offset)
                    drone.in_battery_port = 0
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'normal','port_no':empty_port["port_no"]}
                    # print(final_pos)
                    return {"position" : final_pos, "action": "land"}
            elif action ==2: #land in bat
                empty_port = self.port.get_empty_battery_port()
                if not empty_port:
                    return {"position" : None, "action": "stay-penalty"} #Trying to add a penalty here instead of stay, to give termination criteria to the model
                else:
                    drone.in_battery_port = 1
                    final_pos = self.get_final_pos(empty_port["position"], drone.offset)
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'battery','port_no':empty_port["port_no"]}
                    return {"position" : final_pos, "action": "land-b"}
            else: #move to hover
                empty_port = self.port.get_empty_hover_status()
                final_pos = self.get_final_pos(empty_port["position"], drone.offset)
                self.port.update_port(drone.port_identification)
                drone.port_identification = {'type':'hover','port_no':empty_port["port_no"]}
                return {"position" : final_pos, "action": "hover"}
        elif status == drone.all_states['in-port'] or status == drone.all_states['battery-port']:
            #takeoff things
            if action ==0:
                return {"action": "stay"}
            elif action ==1: #takeoff
                dest = self.port.get_destination(choice=0)
                final_pos = self.get_final_pos(dest, drone.offset)
                self.port.update_port(drone.port_identification)
                return {"position" : final_pos, "action": "takeoff"}
            elif action ==2: #move to bat
                empty_port = self.port.get_empty_battery_port()    
                if not empty_port:
                    empty_port = self.port.get_empty_hover_status()
                    final_pos = self.get_final_pos(empty_port['position'],drone.offset)
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'hover','port_no':empty_port["port_no"]}
                    return {'position':final_pos, 'action':'takeoff-hover'} #Trying to add a penalty here instead of takeoff-hover, to give termination criteria to the model
                else:
                    final_pos = self.get_final_pos(empty_port["position"], drone.offset)
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'battery','port_no':empty_port["port_no"]}
                    return {"position" : final_pos, "action": "move-b"}
            else: #move from bat 
                empty_port = self.port.get_empty_port()
                if not empty_port:
                    empty_port = self.port.get_empty_hover_status()
                    final_pos = self.get_final_pos(empty_port['position'],drone.offset)
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'hover','port_no':empty_port["port_no"]}
                    return {'position':final_pos, 'action':'takeoff-hover'} #Trying to add a penalty here instead of takeoff-hover, to give termination criteria to the model
                else:
                    final_pos = self.get_final_pos(empty_port["position"], drone.offset)
                    self.port.update_port(drone.port_identification)
                    drone.port_identification = {'type':'normal','port_no':empty_port["port_no"]}
                    drone.in_battery_port = 0
                    return {"position" : final_pos, "action": "move"}
        elif status == drone.all_states['in-action']:
            if action == 0: #Stay on path
                return {'action':'continue'}
            elif action == 1: #Deviate from path
                return {'action':'deviate'}
            else: #Any other action should give a penalty while drone is in action, won't happen due to masking 
                return {'action':'reward-penalty'}



    
    def get_final_pos(self,port, offset):
        return [port[0] + offset[0] , port[1] + offset[1], offset[2]]
        
    
    
"""
All actions for reference
LAnding
1.1 stay still - 0
1.2 land in empty port - 1
1.3 land in battery port - 2
1.4 move to empty hovering spot - 3

Takeoff
2.1 stay still - 0
2.2 takeoff - 1
2.3 move to battery port - 2
2.4 move from battery port - 3


"""