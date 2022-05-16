from weakref import finalize
import airsim
import cv2
import numpy as np
import os
import pprint
# import setup_path 
import tempfile

from requests import get

def get_final_pos(port, offset):
        return [port[0] + offset[0] , port[1] + offset[1], offset[2]]

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, "Drone0")

client.armDisarm(True, "Drone0")


# airsim.wait_key('Press any key to takeoff')
f1 = client.takeoffAsync(vehicle_name="Drone0")
# f1 = client.hoverAsync(vehicle_name="Drone1")
f1.join()

# port = [-2,3,-2]
# port = [-1,-9,0]
# port = [-9,-9,0]
# port = [-10,0,0]
# port = [-5,12,0]
# port = [-8,12,0]
# port = [-1,-5,0]
port = [-5,-9,0]


offset = [0,0,-2]
final_pos = get_final_pos(port,offset)


f1 = client.moveToPositionAsync(final_pos[0], final_pos[1], final_pos[2], 1, vehicle_name="Drone0")
f1.join()

# airsim.wait_key('Press any key to land')
# fl = client.landAsync(vehicle_name='Drone1')
f1 = client.hoverAsync(vehicle_name="Drone0")
f1.join()

#hover spots
#-7,-4,0
#-9,-4,0
#-13,0,0
#-11,3,0
#-5,8,0
#-8,8,0
#-4,4,0
#-11,8,0
#Destinations
#21,3,0 North
#-30,3,0 South
#-8,23,0 East
#-8,-17,0 West
#-30,23,0 South East


state1 = client.getMultirotorState(vehicle_name="Drone0")
s = pprint.pformat(state1)
print("state: %s" % s)

# client.armDisarm(False, "Drone1")
# client.armDisarm(False, "Drone2")
# client.reset()

# # that's enough fun for now. let's quit cleanly
# client.enableApiControl(False, "Drone1")
# client.enableApiControl(False, "Drone2")


