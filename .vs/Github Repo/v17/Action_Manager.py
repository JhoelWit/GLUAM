# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:38:39 2022

@author: Prajit
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
        status = drone.get_status()
        if status == 0:
            #here we are in landing phase, return 0 - 4 stay still, land, battery port, empty hovering spot
            if action ==0:
                return {"action": "stay", "position" : None}
            elif action ==1:
                empty_port = self.port.get_empty_port()
                self.port.change_status_normal_port(empty_port["port_no"], True)
                final_pos = self.get_final_pos(empty_port["position"], drone.offset)
                return {"position" : final_pos, "action": "land"}
            elif action ==2:
                empty_port = self.port.get_empty_battery_port()
                self.port.change_status_battery_port(empty_port["port_no"], True)
                final_pos = self.get_final_pos(empty_port["position"], drone.offset)
                print(final_pos)
                return {"position" : final_pos, "action": "land"}
            else:
                empty_port = self.port.get_empty_hover_Spot()
                final_pos = self.get_final_pos(empty_port["position"], drone.offset)
                self.port.hover_spot_status(empty_port["port_no", True])
                return {"position" : final_pos, "action": "land"}
        elif status == 1:
            #takeoff things
            if action ==0:
                return {"action": "stay"}
            elif action ==1:
                dest = self.port.get_destination()
                final_pos = self.get_final_pos(dest, drone.offset)
                return {"position" : final_pos, "action": "takeoff"}
            elif action ==2:
                empty_port = self.port.get_empty_battery_port()             
                final_pos = self.get_final_pos(empty_port, drone.offset)
                self.port.change_status_battery_port(empty_port["port_no"], True)
                return {"position" : final_pos, "action": "move"}
            else:
                empty_port = self.port.get_empty_port()
                self.port.change_status_normal_port(empty_port["port_no"], True)
                return {"position" : final_pos, "action": "move"}
        return None
    
    def get_final_pos(self,port, offset):
        # [x1,y1] , [x2, y2]
        #return [x1+x2, y1+y2]
        pass
    
    
"""
All actions for reference
LAnding
1.1 stay still - 0
1.2 land in empty port - 1
1.3 land in battery port - 2
1.4 move to empty hovering spot – 3

Takeoff
2.1 stay still - 0
2.2 takeoff - 1
2.3 move to battery port - 2
2.4 move from battery port - 3


"""