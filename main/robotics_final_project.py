import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.append("../")

from src.utils import *
from src.core.pybullet_core import PybulletCore

from neuromeka import IndyDCP3
from neuromeka import JointTeleopType, TaskTeleopType

import cv2
import imutils

import camera_detector
import threading

indy = IndyDCP3(robot_ip='192.168.0.22', index=0)

def bounce(robot: IndyDCP3, current_robot_pos: np.ndarray, target_ball_pos: np.ndarray, bounce_force ):
    robot.movetelel_abs(tpos = current_robot_pos - bounce_force * np.array([0, 0, 10, 0, 0, 0]))
    time.sleep(0.05 * bounce_force)
    robot.movetelel_abs(tpos = current_robot_pos)
    time.sleep(0.05 * bounce_force)

#move_response = indy.movej(jtarget = [-27, -20, -74, 98, -27, -113])

indy.stop_teleop()

horizontal_orientation_xyz = np.array([90, 104.5, 90])
home_pos = np.array([500, 10, 500, 95, 104.5, 90]) # x, y, z (mm), x, y, z (deg)
init_jpos = indy.get_control_data()['q']
indy.movel(ttarget = home_pos)

while (indy.get_motion_data()['is_in_motion'] == True):
    time.sleep(0.01)
time.sleep(1)

while (indy.get_control_data()['op_state'] != 17):
    time.sleep(1)
    indy.start_teleop(method=TaskTeleopType.ABSOLUTE)
time.sleep(1)

# initiate camera system
ball_pos = [0, 0, 0]
ball_vel = [0, 0, 0]

camThread = threading.Thread(target = camera_detector, args=())
camThread.start()

'''
repeat = 5
i=0
while i <repeat :
    #bounce(indy, current_robot_pos=home_pos, target_ball_pos = home_pos + np.array([0, +200, 0, 0, 0, 0]), bounce_force = 20)
    #indy.movetelel_abs(home_pos + np.array([0, +200, 0, 0, 0, 0]), vel_ratio=0.5, acc_ratio=1.0)
    #time.sleep(3)
    #bounce(indy, current_robot_pos = home_pos, target_ball_pos = home_pos + np.array([0, -200, 0, 0, 0, 0]))
    #indy.movetelel_abs(home_pos + np.array([0, -200, 0, 0, 0, 0]), vel_ratio=0.5, acc_ratio=1.0)


    indy.movetelel_abs(home_pos, vel_ratio= 0.9, acc_ratio=0.8)
    time.sleep(1)
    indy.movetelel_abs(home_pos + np.array([0, 0, -200, 0, 0, 0]), vel_ratio=0.9, acc_ratio=0.8)
    time.sleep(1)
    
    i = i + 1
'''
time.sleep(3)
indy.stop_teleop()
