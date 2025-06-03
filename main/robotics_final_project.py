import os
import sys

import numpy as np
from math import sqrt, atan2, degrees, radians


import matplotlib.pyplot as plt
import time

sys.path.append("../")

from src.utils import *
from src.core.pybullet_core import PybulletCore

from neuromeka import IndyDCP3
from neuromeka import JointTeleopType, TaskTeleopType

import cv2
import imutils

from utils.Toolbox import *
from utils.Camera.realsense import RealSense, D455_DEFAULT_COLOR, D455_DEFAULT_DEPTH, L515_DEFAULT_DEPTH, L515_DEFAULT_COLOR

import threading
import keyboard


racket_pos = np.array([0, 0, 0])
racket_vel = np.array([0, 0, 0])
transform_matrix = np.eye(4)
ball_pos = np.array([0, 0, 0]) # in mm
ball_vel = np.array([0, 0, 0]) # in mm/s
is_cam_setup = False
stop_camera = False



def calculate_affine_transform(robot_points, camera_points):
    global transform_matrix
    """
    로봇 좌표계 -> 카메라 좌표계로 변환하는 3D 아핀 변환 행렬 계산
    :param robot_points: 로봇 좌표계 점들 [(x1, y1, z1), (x2, y2, z2), ...] (8개 점)
    :param camera_points: 카메라 좌표계 대응점들 [(x1, y1, z1), (x2, y2, z2), ...] (8개 점)
    :return: 4x4 아핀 변환 행렬
    """
    # 동차 좌표계로 변환 (로봇 좌표계)
    A = []
    for x, y, z in camera_points:
        A.append([x, y, z, 1])
    A = np.array(A)  # 8x4 행렬
    
    # 카메라 좌표계 (타겟)
    B = np.array(robot_points)  # 8x3 행렬
    
    # 최소 제곱법으로 변환 행렬 계산 (A * M = B)
    M, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
    
    # 4x4 변환 행렬 구성
    T = np.eye(4)
    T[:3, :] = M.T  # 상위 3x4 부분 적용

    np.savetxt('file_transform_matrix.txt', T, fmt='%.6f')

    transform_matrix = T
    

def transform_point(point):
    """
    변환 행렬을 사용해 점 변환
    :param T: 4x4 변환 행렬
    :param point: 변환할 점 (x, y, z)
    :return: 카메라 좌표계의 점 (x, y, z)
    """
    homogeneous_point = np.array([point[0], point[1], point[2], 1])
    transformed = transform_matrix @ homogeneous_point
    return transformed[:3]  # 동차 좌표 -> 3D 좌표





########################### vision code ###########################


SERIAL1 = "138322252637"
SERIAL2 = "138322250508"

def nothing(x):
    pass

def runCamera():
    global ball_pos
    global ball_vel
    global is_cam_setup

    cam1 = RealSense(serial=SERIAL1)
    cam1.initialize(resolution_color=D455_DEFAULT_COLOR, resolution_depth=D455_DEFAULT_DEPTH)

    cam2 = RealSense(serial=SERIAL2)
    cam2.initialize(resolution_color=D455_DEFAULT_COLOR, resolution_depth=D455_DEFAULT_DEPTH)

    ballLower = (5, 100, 200)
    ballUpper = (78, 255, 255)

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE) #UI

    state_filtered = None # [x, y, r, xc, xy]
    state_filtered1 = None
    state_filtered2 = None
    z_filtered = None
    beta = 0.80
    ball_diameter = 0.05 # 4cm ball
    state_filtered_que = [None] * 30
    prev_pos = None
    last_time = time.time()

    # num_memory = 1 # need to implement?

    is_cam_setup = True
    #print(cam1._fx, cam1._fy)
    while True:
        ts = time.time_ns()

        # img_rgb1, map_depth1, img_depth1 = cam1.get_color_depth('cv', clipping_depth=2.0)
        img_rgb1, map_depth1, img_depth1 = cam1.get_color_depth('cv')
        # img_rgb1 = cv2.resize(img_rgb1, dsize=(320, 180), interpolation=cv2.INTER_AREA)

        img_rgb2, map_depth2, img_depth2 = cam2.get_color_depth('cv')

        img_blur1 = cv2.GaussianBlur(img_rgb1, (11, 11), 0)
        img_hsv1 = cv2.cvtColor(img_blur1, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(img_hsv1, ballLower, ballUpper)
        mask1 = cv2.erode(mask1, None, iterations=2)
        mask1 = cv2.dilate(mask1, None, iterations=2)

        img_blur2 = cv2.GaussianBlur(img_rgb2, (11, 11), 0)
        img_hsv2 = cv2.cvtColor(img_blur2, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(img_hsv2, ballLower, ballUpper)
        mask2 = cv2.erode(mask2, None, iterations=2)
        mask2 = cv2.dilate(mask2, None, iterations=2)

        cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts1 = imutils.grab_contours(cnts1)

        cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = imutils.grab_contours(cnts2)

        center = None

        # only proceed if at least one contour was found
        if len(cnts1) > 0:
            # find the largest contour in the mask1, then use it to compute the minimum enclosing circle and centroid

            # u,v,w are coordinates in image frame and x,y,z are coordinates in real world. related by scalar multiplication cam._fx and axis transformation
            # cam1._fx = 628.456298828125, cam1._fy = 627.6700439453125
            # u // y
            # v // -z
            # w // -x
            # x // -w
            # y // u
            # z // -v
            
            c1 = max(cnts1, key=cv2.contourArea)

            ((u1, v1), r1) = cv2.minEnclosingCircle(c1)

            ''' # maybe redundant
            x1 = u1 / cam1._fx
            y1 = v1 / cam1._fx
            ''' # maybe redundant

            M1 = cv2.moments(c1)
            uc1 = M1["m10"] / M1["m00"]
            vc1 = M1["m01"] / M1["m00"]
            wc1 = ball_diameter / (r1 * 2 / cam1._fx)

            ''' # maybe redundant
            xc1 = - wc1
            yc1 = uc1 / cam1._fx
            zc1 = - vc1 / cam1._fx
            ''' # maybe redundant

            state_observed1 = np.array([uc1, vc1, wc1, r1,0, 0, 0])

            # only proceed if the radius meets a minimum size
            if r1 > 8:
                if state_filtered1 is None:
                    state_filtered1 = state_observed1.copy()

                state_filtered1 = beta * state_observed1 + (1-beta) * state_filtered1
            else:
                state_filtered1 = state_observed1
        else:
            if state_filtered1 is None:
                state_filtered1 = np.zeros(7)
        
        if len(cnts2) > 0:
            # find the largest contour in the mask1, then use it to compute the minimum enclosing circle and centroid

            # u,v,w are coordinates in image frame and x,y,z are coordinates in real world. related by scalar multiplication cam._fx and axis transformation
            # cam1._fx = 628.456298828125, cam1._fy = 627.6700439453125
            # u // y
            # v // -z
            # w // -x
            # x // -w
            # y // u
            # z // -v
            
            c2 = max(cnts2, key=cv2.contourArea)

            ((u2, v2), r2) = cv2.minEnclosingCircle(c2)

            ''' # maybe redundant
            x2 = u2 / cam2._fx
            y2 = v2 / cam2._fx
            ''' # maybe redundant

            M2 = cv2.moments(c2)
            uc2 = M2["m10"] / M2["m00"]
            vc2 = M2["m01"] / M2["m00"]
            wc2 = ball_diameter / (r2* 2 / cam2._fx)

            ''' # maybe redundant
            xc2 = - wc2
            yc2 = uc2 / cam2._fx
            zc2 = - vc2 / cam2._fx
            ''' # maybe redundant

            vc2=vc2-46.8
            uc2=uc2-19
            state_observed2 = np.array([uc2, vc2, wc2, r2, 0,0,0])

            # only proceed if the radius meets a minimum size
            if r2 > 8:
                if state_filtered2 is None:
                    state_filtered2 = state_observed2.copy()

                state_filtered2 = beta * state_observed2 + (1-beta) * state_filtered2
            else:
                state_filtered2 = state_observed2
        else:
            if state_filtered2 is None:
                state_filtered2 = np.zeros(7)
        # state_observed1 = np.array([uc1, vc1, wc1, r1, xc1, yc1, zc1])=[y모멘텀, z모멘텀, x모멘텀, 반지름, x]=
        state_filtered=np.array([state_filtered1[0],(state_filtered1[1]+state_filtered2[1])/2,state_filtered2[0],state_filtered1[3], 0,0,0])
        state_filtered[4] = - state_filtered[2] / cam1._fx
        state_filtered[5] = state_filtered[0] / cam1._fx
        state_filtered[6] = - state_filtered[1] / cam1._fx

        
        # Update Ball Position
        ball_pos = transform_point([state_filtered[0], state_filtered[2], state_filtered[1]])

        state_filtered_que.append(state_filtered)
        if len(state_filtered_que) > 30:
            state_filtered_que.pop(0)

        try:
            curr_time = time.time()
            if (prev_pos is None):
                last_time = curr_time
                prev_pos = ball_pos
            else :
                if ((curr_time - last_time) > 0.035):
                    ball_vel = (ball_pos - prev_pos) / (curr_time - last_time)
                    last_time = curr_time
                    prev_pos = ball_pos
        except:
            ball_vel = np.array([0, 0, 0])



        # UI
        #노란색 원 코드        
        cv2.circle(img_rgb1, (int(state_filtered[0]), int(state_filtered[1])), int(state_filtered[3]), (0, 255, 255), 2)

        #빨간색 점 코드
        for j in range(1, len(state_filtered_que)):
            if state_filtered_que[j - 1] is None or state_filtered_que[j] is None:
                continue
            pt1 = [int(x) for x in state_filtered_que[j - 1][0:2]]
            pt2 = [int(x) for x in state_filtered_que[j][0:2]]
            cv2.line(img_rgb1, pt1, pt2, (0, 0, 255), int(2 + 5 / float(len(state_filtered_que) - j)))

        tf = time.time_ns()

        if (tf-ts)/1e9 > 0.01:
            cv2.putText(img_rgb1, '{0:02.1f} FPS'.format(1/((tf-ts)/1e9)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 255, 255), 10, cv2.LINE_AA)
            cv2.putText(img_rgb1, '{0:02.1f} FPS'.format(1/((tf-ts)/1e9)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 0), 5, cv2.LINE_AA)

        output1 = np.vstack((img_rgb1, img_depth1))
        output1 = cv2.resize(output1, dsize=(640, 720), interpolation=cv2.INTER_AREA)
        output2 = np.vstack((img_rgb2, img_depth2))
        output2 = cv2.resize(output2, dsize=(640, 720), interpolation=cv2.INTER_AREA)
        combined_output=np.hstack((output1,output2))
        cv2.imshow('RealSense', combined_output)

        k = cv2.waitKey(1) & 0xFF
        

        if (stop_camera):
            cv2.destroyAllWindows() # UI
            break



########################### /vision code ###########################







indy = IndyDCP3(robot_ip='192.168.0.22', index=0)

def print_coordinate():
    print("x: " + str(ball_pos[0]) + " y: " + str(ball_pos[1]) + " z: " + str(ball_pos[2]))

def robot_calibration(home_pos : np.ndarray, w, h):
    print("#### starting calibration... ####")
    time_per_step = 3
    vel_ratio = 0.3
    acc_ratio = 1.0

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    target_points = [
        (w/2, w/2, h/2),  # 1
        (-w/2, w/2, h/2),  # 2
        (-w/2, -w/2, h/2),  # 3
        (w/2, -w/2, h/2),  # 4
        (w/2, w/2, -h/2),  # 5
        (-w/2, w/2, -h/2),  # 6
        (-w/2, -w/2, -h/2),  # 7
        (w/2, -w/2, -h/2)   # 8
    ]
    camera_points = []

    indy.movetelel_abs(home_pos, vel_ratio=0.9, acc_ratio=0.8)
    time.sleep(1)
    print("home ", end='')
    print_coordinate()
    ax.scatter(ball_pos[0], ball_pos[1], ball_pos[2])

    indy.movetelel_abs(home_pos + np.array([+w/2, +w/2, +h/2, 0, 0, 0]), vel_ratio, acc_ratio)
    time.sleep(time_per_step + 0.5)
    print("pos #1 ", end='')
    print_coordinate()
    camera_points.append((ball_pos[0], ball_pos[1], ball_pos[2]))
    ax.scatter(ball_pos[0], ball_pos[1], ball_pos[2])

    indy.movetelel_abs(home_pos + np.array([-w/2, +w/2, +h/2, 0, 0, 0]), vel_ratio, acc_ratio)
    time.sleep(time_per_step)
    print("pos #2 ", end='')
    print_coordinate()
    camera_points.append((ball_pos[0], ball_pos[1], ball_pos[2]))
    ax.scatter(ball_pos[0], ball_pos[1], ball_pos[2])

    indy.movetelel_abs(home_pos + np.array([-w/2, -w/2, +h/2, 0, 0, 0]), vel_ratio, acc_ratio)
    time.sleep(time_per_step)
    print("pos #3 ", end='')
    print_coordinate()
    camera_points.append((ball_pos[0], ball_pos[1], ball_pos[2]))
    ax.scatter(ball_pos[0], ball_pos[1], ball_pos[2])

    indy.movetelel_abs(home_pos + np.array([+w/2, -w/2, +h/2, 0, 0, 0]), vel_ratio, acc_ratio)
    time.sleep(time_per_step)
    print("pos #4 ", end='')
    print_coordinate()
    camera_points.append((ball_pos[0], ball_pos[1], ball_pos[2]))
    ax.scatter(ball_pos[0], ball_pos[1], ball_pos[2])

    indy.movetelel_abs(home_pos + np.array([+w/2, +w/2, -h/2, 0, 0, 0]), vel_ratio, acc_ratio)
    time.sleep(time_per_step)
    print("pos #5 ", end='')
    print_coordinate()
    camera_points.append((ball_pos[0], ball_pos[1], ball_pos[2]))
    ax.scatter(ball_pos[0], ball_pos[1], ball_pos[2])

    indy.movetelel_abs(home_pos + np.array([-w/2, +w/2, -h/2, 0, 0, 0]), vel_ratio, acc_ratio)
    time.sleep(time_per_step)
    print("pos #6 ", end='')
    print_coordinate()
    camera_points.append((ball_pos[0], ball_pos[1], ball_pos[2]))
    ax.scatter(ball_pos[0], ball_pos[1], ball_pos[2])

    indy.movetelel_abs(home_pos + np.array([-w/2, -w/2, -h/2, 0, 0, 0]), vel_ratio, acc_ratio)
    time.sleep(time_per_step)
    print("pos #7 ", end='')
    print_coordinate()
    camera_points.append((ball_pos[0], ball_pos[1], ball_pos[2]))
    ax.scatter(ball_pos[0], ball_pos[1], ball_pos[2])

    indy.movetelel_abs(home_pos + np.array([+w/2, -w/2, -h/2, 0, 0, 0]), vel_ratio, acc_ratio)
    time.sleep(time_per_step)
    print("pos #8 ", end='')
    print_coordinate()
    camera_points.append((ball_pos[0], ball_pos[1], ball_pos[2]))
    ax.scatter(ball_pos[0], ball_pos[1], ball_pos[2])

    calculate_affine_transform(target_points, camera_points)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #print("#### check points with plot figure ####")
    #plt.show()

    np.savetxt('file_cam_point.txt', camera_points, fmt='%.6f')



############### test ###############
def get_racket_launch_delay_after_peak(ball_z_max_pos, impact_z, racket_up_time, g=9810):
    """
    공이 최고점에 도달한 이후, 몇 초 후에 라켓을 움직이기 시작해야 하는지를 반환합니다.
    """
    dz = ball_z_max_pos - impact_z
    if dz < 0:
        print("impact_z는 ball_z_max_pos보다 낮아야 합니다.")
        return 0

    t_to_impact = sqrt(2 * dz / g)
    return t_to_impact - racket_up_time


def compute_racket_orientation(target_z, restitution=0.8, g=9810):  # g in mm/s²
    global ball_pos, ball_vel, racket_vel

    pos = np.array(ball_pos)        # mm
    vel_in = np.array(ball_vel)     # mm/s
    v_racket_z = racket_vel[2]      # mm/s

    # 목표 위치 변경: (0, 10, target_z)
    target = np.array([-5.0, 15.0, target_z])
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    delta_z = target_z - pos[2]

    # z축 상대속도 및 반사 후 속도
    vz_rel_in = vel_in[2] - v_racket_z
    vz_rel_out = -restitution * vz_rel_in
    vz_out = vz_rel_out + v_racket_z

    # 포물선 도달 시간 계산 (z 방향 2차방정식 해)
    a = -0.5 * g
    b = vz_out
    c = delta_z

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        t_flight = 0.1  # fallback
    else:
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        t_flight = max(t1, t2) if max(t1, t2) > 0 else 0.1

    # x, y 방향 속도 계산
    vx_out = dx / t_flight
    vy_out = dy / t_flight
    vel_out = np.array([vx_out, vy_out, vz_out])

    # surface normal 계산
    n = vel_in - vel_out
    n = n / np.linalg.norm(n)

    # roll = x축 회전, pitch = y축 회전
    roll  = np.arcsin(-n[0])  # x축 회전
    pitch = np.arcsin(n[1])   # y축 회전

    return roll, pitch



def compute_parabolic_roll_pitch(x, y, width, max_degree, para_z, base_z):
    half = width / 2
    x_norm = np.clip(x / half, -1.0, 1.0)
    y_norm = np.clip(y / half, -1.0, 1.0)

    delta_z = para_z - base_z
    if delta_z <= 0:
        raise ValueError("para_z must be greater than base_z.")
    a = - 1 / (4 * delta_z)

    x_real = x_norm * half
    y_real = y_norm * half

    dzdx = 2 * a * x_real
    dzdy = 2 * a * y_real

    n = np.array([-dzdx, -dzdy, 1.0])
    n /= np.linalg.norm(n)

    roll_rad = np.arcsin(n[1])
    pitch_rad = np.arcsin(-n[0])

    max_rad = radians(max_degree)
    roll_rad = np.clip(roll_rad, -max_rad, max_rad)
    pitch_rad = np.clip(pitch_rad, -max_rad, max_rad)

    return roll_rad, pitch_rad




def compute_gaussian_roll_pitch(x, y, amplitude=100, sigma=800):
    A = amplitude
    r2 = x**2 + y**2
    exp_part = np.exp(-r2 / (2 * sigma**2))

    dz_dx = (A * x / sigma**2) * exp_part
    dz_dy = (A * y / sigma**2) * exp_part

    roll_rad = np.arctan(dz_dy)
    pitch_rad = -np.arctan(dz_dx)

    return roll_rad, pitch_rad




def compute_linear_roll_pitch(x, y, width, max_degree): # deprecated. useless
    # 정규화: -1.0 ~ 1.0 범위
    half = width / 2
    x_norm = np.clip(x / half, -1.0, 1.0)
    y_norm = np.clip(y / half, -1.0, 1.0)

    roll_rad = np.radians(y_norm * max_degree)
    pitch_rad = np.radians(-x_norm * max_degree)

    return roll_rad, pitch_rad



def euler_xyz_to_matrix(euler_deg):
    x, y, z = np.radians(euler_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])
    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])
    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def matrix_to_euler_xyz(R):
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    if abs(R[0, 2]) < 1.0:
        x = np.atan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.atan2(-R[1,2], R[1,1])
        y = np.atan2(-R[2,0], sy)
        z = 0.0

    return np.degrees([x, y, z])

def apply_roll_pitch(original_angles_deg, roll_rad, pitch_rad):
    R_orig = euler_xyz_to_matrix(original_angles_deg)

    # roll (X), pitch (Y)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad),  np.cos(roll_rad)]
    ])

    Ry = np.array([
        [ np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])

    R_final = Ry @ Rx @ R_orig

    return matrix_to_euler_xyz(R_final)










########################### main code ###########################

indy.stop_teleop()

home_pos = np.array([595, 25, 420, -89.231, 76.476, -94.132]) # x, y, z (mm), x, y, z (deg)
indy.movel(ttarget = home_pos)

#init_jpos = indy.get_control_data()['q']

while (indy.get_motion_data()['is_in_motion'] == True):
    time.sleep(0.01)
time.sleep(1)

# set the camera mode to teleop abs
while (indy.get_control_data()['op_state'] != 17):
    indy.start_teleop(method=TaskTeleopType.ABSOLUTE)
    time.sleep(1)
time.sleep(1)

# initiate camera system
camThread = threading.Thread(target = runCamera)
camThread.start()

while (not is_cam_setup) :
    time.sleep(0.5)
    #print("waiting for cam setup")

print("#### cam setup done! ####")
time.sleep(5)

try:
    workspace_width = 350
    workspace_height = 300
    workspace_tolerance = 10
    robot_calibration(home_pos, workspace_width, workspace_height)
    time.sleep(2)
    indy.movetelel_abs(home_pos, 0.4, 1.0)
    time.sleep(2)

    vel_threshold = 100 # in mm/s
    target_z = -workspace_height/2
    orientation_lock = False
    bounce_height = 80
    test_roll_rad = 0
    test_pitch_rad = 0

    pos_data = open("file_pos_data.txt", 'w')
    vel_data = open("file_vel_data.txt", 'w')
    time_data = open("file_time_data.txt", 'w')
    robot_pos_data = open("file_robot_pos_data.txt", 'w')
    robot_vel_data = open("file_robot_vel_data.txt", 'w')
    roll_pitch_data = open("file_roll_pitch_data.txt", 'w')

    start_time = time.time()
    count = 0

    max_ball_z_pos = 0
    max_ball_z_time = time.time()
    is_ball_falling = False

    print("#### detach the ball and press s ####")
    while (True):
        if (keyboard.is_pressed('s')):
            break
        time.sleep(0.1)

    while (True):
        if (keyboard.is_pressed('esc')):
            break

        robot_pos_raw = indy.get_control_state()['p']
        racket_pos = np.array([robot_pos_raw[0] - home_pos[0], robot_pos_raw[1] - home_pos[1], robot_pos_raw[2] - home_pos[2]])
        racket_vel = indy.get_control_state()['pdot']
        robot_pos_data.write(str(racket_pos[0]) + "," + str(racket_pos[1]) + "," + str(racket_pos[2]) + "\n")
        robot_vel_data.write(str(racket_vel[0]) + "," + str(racket_vel[1]) + "," + str(racket_vel[2]) + "\n")
        
        now_time = time.time()
        pos_data.write(str(ball_pos[0]) + "," + str(ball_pos[1]) + "," + str(ball_pos[2]) + "\n")
        vel_data.write(str(ball_vel[0]) + "," + str(ball_vel[1]) + "," + str(ball_vel[2]) + "\n")
        time_data.write(str(now_time) + "\n")

        count = count + 1
        if (now_time - start_time > 1):
            print("sampling per second : " + str(count))
            count = 0
            start_time = now_time


        if ((np.abs(ball_pos[0]) > (workspace_width/2 + workspace_tolerance))
            or (np.abs(ball_pos[1]) > (workspace_width/2 + workspace_tolerance))):
            roll_pitch_data.write("0.0,0.0\n")
            print("#### ball is out of workspace. esc to stop, s to start again ####")
            selection = 'esc'
            while (True):
                if (keyboard.is_pressed('esc')):
                    selection = 'esc'
                    break
                elif (keyboard.is_pressed('s')):
                    selection = 's'
                    break
                time.sleep(0.1)
            if (selection == 's'):
                continue
            else:
                break
        
        
        if (is_ball_falling):
            orientation_lock = False
            when_to_up = max_ball_z_time + get_racket_launch_delay_after_peak(max_ball_z_pos, -workspace_height/2 + bounce_height, 0.45)
            if (now_time >= when_to_up):
                orientation_lock = False
                target_z = -workspace_height/2 + bounce_height + 10
            if (ball_vel[2] > vel_threshold):
                target_z = -workspace_height/2
                is_ball_falling = False
        else :
            orientation_lock = False
            test_roll_rad = 0
            test_pitch_rad = 0
            if (ball_vel[2] >= 0):
                if (ball_pos[2] > max_ball_z_pos):
                    max_ball_z_pos = ball_pos[2]
                    max_ball_z_time = now_time
            elif (ball_pos[2] > max_ball_z_pos):
                max_ball_z_pos = ball_pos[2]
                max_ball_z_time = now_time
            elif (ball_vel[2] < -vel_threshold):
                is_ball_falling = True
        


        #roll_rad, pitch_rad = compute_linear_roll_pitch(ball_pos[0], ball_pos[1], workspace_width, 10)
        if (not orientation_lock):
            #test_roll_rad, test_pitch_rad = compute_linear_roll_pitch(ball_pos[0], ball_pos[1], workspace_width, 10)
            #test_roll_rad, test_pitch_rad = compute_parabolic_roll_pitch(ball_pos[0], ball_pos[1], workspace_width, 20, 1500, -workspace_height/2 + bounce_height)
            test_roll_rad, test_pitch_rad = compute_gaussian_roll_pitch(ball_pos[0], ball_pos[1])
            '''
            test_roll_rad, test_pitch_rad = compute_racket_orientation(-workspace_height/2 + bounce_height) #racket_pos[2] ? -workspace_height/2 + bounce_height ? redundant???
            if (test_roll_rad > 0.5): # ~= +-30 deg
                test_roll_rad = 0.5
            elif (test_roll_rad < -0.5):
                test_roll_rad = -0.5
            if (test_pitch_rad > 0.5): # ~= +-30 deg
                test_pitch_rad = 0.5
            elif (test_pitch_rad < -0.5):
                test_pitch_rad = -0.5
            '''

        roll_pitch_data.write(str(np.degrees(test_roll_rad)) + "," + str(np.degrees(test_pitch_rad)) + "\n")

        #print("Ref. value : " + str(np.degrees(roll_rad)) + "," + str(np.degrees(pitch_rad)) + " Test value : " + str(test_roll_deg) + "," + str(test_pitch_deg))
        lacket_angles = apply_roll_pitch(home_pos[3:6], test_roll_rad, test_pitch_rad)

        indy.movetelel_abs(np.array([home_pos[0] + ball_pos[0], home_pos[1] + ball_pos[1], home_pos[2] + target_z,
                                    lacket_angles[0], lacket_angles[1], lacket_angles[2]]), vel_ratio=1.0, acc_ratio=2.0)
        
    
except Exception as e:
    print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

finally:
    indy.stop_teleop()
    stop_camera = True
    pos_data.close()
    vel_data.close()
    time_data.close()
    robot_pos_data.close()
    robot_vel_data.close()
    roll_pitch_data.close()

    time.sleep(2)
    indy.get_violation_data()
    indy.recover()
