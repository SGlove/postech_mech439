import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import time
import pyrealsense2 as rs

from neuromeka import IndyDCP3
from src.core.pybullet_core import PybulletCore
from utils.Toolbox import *
from utils.Camera.realsense import RealSense, D455_DEFAULT_COLOR, D455_DEFAULT_DEPTH, L515_DEFAULT_DEPTH, L515_DEFAULT_COLOR

# serial number of each camera
SERIAL1 = "138322252637"
SERIAL2 = "138322250508"

# pixel width and height of D455 camera screen
pixel_width = 1280 # pixel
pixel_height = 720 # pixel

# AOV (angle of view) of D455
#

# physcial information of ping-pong lacket
neck_length = 10 + 26 # mm
pingpong_r = 150 # mm
thickness = 10 # mm

def nothing(x):
    pass

# function conducting euler angle to rotation matrix conversion
def euler_2_rotation_ZYX(phi, theta, psi):
    R_Z = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])

    R_Y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    R_X = np.array([
        [1, 0, 0],
        [0, np.cos(phi), np.sin(phi)],
        [0, -np.sin(phi), np.cos(phi)]
    ])

    R = R_Z @ R_Y @ R_X
    return R

def get_ray_direction(u, v, fx, fy, cx, cy, R):
    x = 1.0
    y = - (u - cx) / fx
    z = - (v - cy) / fy

    ray_cam = np.array([x, y, z], dtype = np.float32).reshape(3,)
    ray_cam = ray_cam / np.linalg.norm(ray_cam)
    
    ray_world = R @ ray_cam
    ray_world = ray_world / np.linalg.norm(ray_world)
    
    return ray_world.reshape(-1)

def triangulate_two_rays(c1, d1, c2, d2): # c: position vector of camera, d: direction vector of ball at each camera
    A = np.stack([d1, d2], axis = 1)
    b = c2 - c1    
    
    t_vals, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    t1, t2 = t_vals

    p1 = c1 + t1 * d1
    p2 = c2 + t2 * d2

    midpoint = (p1 + p2) / 2

    return midpoint

# Kalman Filter class
class KalmanFilter:
    def __init__(self, num_memory = 1): # num_memory: 몇 개의 객체를 추적할 지

        self.M = num_memory
        self.dt = np.float32(1/100) # dt: 시간 간격

        self.kf = cv2.KalmanFilter(6*self.M, 6*self.M, 1) # kalmanfilter(size of state vector(x,y,z,vx,vy,vz) = 6, size of measurement vector = 6, size of contorl vector = 1)
        # self.kf.measurementMatrix = np.zeros([3 * self.M, 6 * self.M], dtype=np.float32)
        self.kf.measurementMatrix = np.eye(6 * self.M, dtype=np.float32)  # eye: diagonal matrix
        self.kf.transitionMatrix  = np.zeros([6 * self.M, 6 * self.M], dtype=np.float32)
        self.kf.controlMatrix     = np.zeros([6 * self.M, 1], dtype=np.float32)

        for m in range(self.M):
            # self.kf.measurementMatrix[3*m:3*(m+1), 6*m:6*(m+1)] = np.array([
            #     [1, 0, 0, 0, 0, 0],
            #     [0, 1, 0, 0, 0, 0],
            #     [0, 0, 1, 0, 0, 0]], np.float32)
            
            self.kf.controlMatrix[6*m:6*(m+1), :] = np.array([
                [0],
                [0],
                [0],
                [0],
                [0],
                [-9.81 * self.dt]], np.float32)

            self.kf.transitionMatrix[6*m:6*(m+1), 6*m:6*(m+1)] = np.array([
                [1, 0, 0, self.dt, 0, 0],
                [0, 1, 0, 0, self.dt, 0],
                [0, 0, 1, 0, 0, self.dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]], np.float32)

        self.kf.errorCovPre = np.identity(6*self.M, np.float32) * 1 # 예측 상태의 오차 공분산 초기값 (초기 신뢰도)
        self.kf.measurementNoiseCov = np.identity(6*self.M, dtype=np.float32) * 3 # 측정 노이즈의 공분산 (3배 크기)

        self.input = np.array([[np.float32(1)]]) # 제어 입력 벡터

    # def correct(self, x, y):
    #     measured = np.array([[np.float32(x)], [np.float32(y)]])
    #     self.kf.correct(measured)
    #     predicted = self.kf.predict(1).reshape(-1)
    #     return predicted

    def predict(self, x, y, z, vx, vy, vz, predict_interval): # predict_interval: 예측하고자 하는 시간 길이 (s)
        measured = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)], [np.float32(vx)], [np.float32(vy)], [np.float32(vz)]])
        statePost = self.kf.correct(measured) # correct: 예측값과 실제 센서 측정값의 차이를 보정하여 필터 상태 최신화
        statePre = self.kf.predict(self.input).reshape(-1) 

        x_pred_list = []
        y_pred_list = []
        z_pred_list = []
        cov_pred_list = []

        statePredict = statePost.copy()
        errorCovPredict = self.kf.errorCovPost.copy()

        x_pred_list.append(statePredict[0, 0])
        y_pred_list.append(statePredict[1, 0])
        z_pred_list.append(statePredict[2, 0])
        cov_pred_list.append(errorCovPredict)
        num_predict = int(predict_interval / self.dt)
        for i in range(num_predict):
            statePredict = self.kf.transitionMatrix @ statePredict + self.kf.controlMatrix @ self.input
            errorCovPredict = self.kf.transitionMatrix @ errorCovPredict + self.kf.transitionMatrix.T @ self.kf.processNoiseCov
            x_pred_list.append(statePredict[0, 0])
            y_pred_list.append(statePredict[1, 0])
            z_pred_list.append(statePredict[2, 0])
            cov_pred_list.append(errorCovPredict)

        return x_pred_list, y_pred_list, z_pred_list, cov_pred_list

# euler angle and position of each camera
euler_cam1 = np.radians([0, 0, 180]) # (phi, theta, psi) degree
pos_cam1 = np.array([115, -4, 56.5 - 2.1]) / 100 # (x, y, z) (cm / 100 = m)
rotation_matrix_cam1 = euler_2_rotation_ZYX(*euler_cam1)

euler_cam2 = np.radians([0, 0, -90]) # (phi, theta, psi) degree
pos_cam2 = np.array([59.5 + 4, 80, 56.5 - 2.1]) / 100 # (x, y, z) (cm / 100 = m)
rotation_matrix_cam2 = euler_2_rotation_ZYX(*euler_cam2)
 
######################### Main Code ###################################
if __name__ == '__main__':
    # connecting each cameras
    cam1 = RealSense(serial = SERIAL1)
    cam1.initialize(resolution_color = D455_DEFAULT_COLOR, resolution_depth = D455_DEFAULT_DEPTH)

    cam2 = RealSense(serial = SERIAL2)
    cam2.initialize(resolution_color = D455_DEFAULT_COLOR, resolution_depth = D455_DEFAULT_DEPTH)
    
    # HSV value for detecting contour
    ball_lower1 = (6, 83, 149)
    ball_upper1 = (20, 237, 255)

    ball_lower2 = (9, 83, 149)
    ball_upper2 = (76, 230, 255)

    ray_1, ray_2 = None, None

    state_filtered = None # [x, y, r, xc, xy]
    state_filtered1 = None
    state_filtered2 = None

    beta = 0.80
    ball_diameter = 0.04 # 4cm ball
    ball_pos = None
    ball_pos_que = [None] * 30

    num_memory = 1
    kalman_filter = KalmanFilter(num_memory)

    # count = 0
    # cam1_error = np.zeros(3)
    # cam2_error = np.zeros(3)

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    
    #############################  for position test #########################
    indy = IndyDCP3(robot_ip='192.168.0.22', index=0)

    indy.recover()

    home_pos = np.array([500, 000, 500, 90, 104.5, 90]) # x, y, z (mm), x, y, z (deg)
    des_pos = np.array([500, 000, 544 - 25 -100, 90, 104.5, 90]) # x, y, z (mm), x, y, z (deg)

    init_jpos = indy.get_control_data()['q']
    indy.movel(ttarget = des_pos)

    real_ball_pos = np.array([home_pos[0] + 115, home_pos[1], home_pos[2] + 5 + 20]) / 1000 # mm
    # print("real ball pose: ", real_ball_pos)
    ##########################################################################

    while True:
        ts = time.time_ns()

        img_rgb1, map_depth1, img_depth1 = cam1.get_color_depth('cv')
        img_rgb2, map_depth2, img_depth2 = cam2.get_color_depth('cv')

        img_blur1 = cv2.GaussianBlur(img_rgb1, (11, 11), 0)
        img_hsv1 = cv2.cvtColor(img_blur1, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(img_hsv1, ball_lower1, ball_upper1)
        mask1 = cv2.erode(mask1, None, iterations=2)
        mask1 = cv2.dilate(mask1, None, iterations=2)

        img_blur2 = cv2.GaussianBlur(img_rgb2, (11, 11), 0)
        img_hsv2 = cv2.cvtColor(img_blur2, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(img_hsv2, ball_lower2, ball_upper2)
        mask2 = cv2.erode(mask2, None, iterations=2)
        mask2 = cv2.dilate(mask2, None, iterations=2)

        cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts1 = imutils.grab_contours(cnts1)

        cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = imutils.grab_contours(cnts2)

        # only proceed if at least one contour was found
        if len(cnts1) > 0 and len(cnts2) > 0:  
            # find the largest contour in the mask1, then use it to compute the minimum enclosing circle and centroid

            c1 = max(cnts1, key=cv2.contourArea)
            c2 = max(cnts2, key=cv2.contourArea)

            ((u1, v1), r1) = cv2.minEnclosingCircle(c1)
            ((u2, v2), r2) = cv2.minEnclosingCircle(c2)

            ray_1 = get_ray_direction(u1, v1, cam1._fx, cam1._fy, pixel_width/2, pixel_height/2, rotation_matrix_cam1)
            ray_2 = get_ray_direction(u2, v2, cam2._fx, cam2._fy, pixel_width/2, pixel_height/2, rotation_matrix_cam2)

            ball_pos = triangulate_two_rays(pos_cam1, ray_1, pos_cam2, ray_2) # np.array([0, 0, 0]) 
            print("meas ball pos: ", ball_pos)
            print("real ball pos: ", real_ball_pos)
            # print("ray1: ", ray_1)

            # M1 = cv2.moments(c1)
            # uc1 = M1["m10"] / M1["m00"]
            # vc1 = M1["m01"] / M1["m00"]

            # ref_meas_distance_1 = 0.478
            # ref_real_distance_1 = 0.508 + 0.02
            # distance_correction_1 = ref_real_distance_1 / ref_meas_distance_1

            # real_distance_1 = np.sqrt((pos_cam1[0] - real_ball_pos[0])**2 + (pos_cam1[1] - real_ball_pos[1])**2 + (pos_cam1[2] - real_ball_pos[2])**2)
            # print("real distance 1: ", real_distance_1)

            # distance_1 = ball_diameter / (r1 * 2 / cam1._fx) # * distance_correction_1

            # print("meas distance 1: ", distance_1)

            # print("coefficient: ", real_distance_1 / distance_1)

            # dy_1 = - (uc1 - pixel_width / 2) / cam1._fx
            # dz_1 = (vc1 - pixel_height / 2) / cam1._fy

            # x_1 = distance_1 / np.sqrt(1 + dy_1**2 + dz_1**2)
            # y_1 = dy_1 * x_1
            # z_1 = dz_1 * x_1

            # state_observed1 = np.array([uc1, vc1, distance_1, r1, x_1, y_1, z_1])

            # only proceed if the radius meets a minimum size
        #     if r1 > 8:
        #         if state_filtered1 is None:
        #             state_filtered1 = state_observed1.copy()

        #         state_filtered1 = beta * state_observed1 + (1-beta) * state_filtered1
        #     else:
        #         state_filtered1 = state_observed1
        # else:
        #     if state_filtered1 is None:
        #         state_filtered1 = np.zeros(7)

        # if len(cnts2) > 0:
            # find the largest contour in the mask1, then use it to compute the minimum enclosing circle and centroid
            


            # print("ray2: ", ray_2)
            
            # M2 = cv2.moments(c2)
            # uc2 = M2["m10"] / M2["m00"]
            # vc2 = M2["m01"] / M2["m00"]

            # ref_meas_distance_2 = 0.53
            # ref_real_distance_2 = 0.792 + 0.02
            
            # distance_correction_2 = ref_real_distance_2 / ref_meas_distance_2

            # distance_2 = ball_diameter / (r2 * 2 / cam2._fx) * distance_correction_2

            # print("distance 2: ", distance_2)

            # dy_2 = - (uc2 - pixel_width / 2) / cam2._fx
            # dz_2 = (vc2 - pixel_height / 2) / cam2._fy

            # x_2 = distance_2 / np.sqrt(1 + dy_2**2 + dz_2**2)
            # y_2 = dy_2 * x_2
            # z_2 = dz_2 * x_2

            # state_observed2 = np.array([uc2, vc2, distance_2, r2, x_2, y_2, z_2])

            # print(state_observed1)

            # only proceed if the radius meets a minimum size   
        #     if r2 > 8:
        #         if state_filtered2 is None:
        #             state_filtered2 = state_observed2.copy()

        #         state_filtered2 = beta * state_observed2 + (1-beta) * state_filtered2
        #     else:
        #         state_filtered2 = state_observed2
        # else:
        #     if state_filtered2 is None:
        #         state_filtered2 = np.zeros(7)

        state_filtered = ball_pos
        state_filtered1 = ball_pos
        state_filtered2 = ball_pos

        # print("ball_pos: ", ball_pos)

        # ball_pos_cam1 = state_filtered1[4:7]
        # ball_pos_cam2 = state_filtered2[4:7]

        # print("for camera 1: ", ball_pos_cam1)
        # print("for camera 2: ", ball_pos_cam2)

        # pos_correction_matrix1 = np.array([0.018, 0.026, -0.052])
        # pos_correction_matrix2 = np.array([-0.017, 0.063, -0.039])

        

        # ball_pos_real_world1 = rotation_matrix_cam1 @ ball_pos_cam1 + pos_cam1 # + pos_correction_matrix1
        # ball_pos_real_world2 = rotation_matrix_cam2 @ ball_pos_cam2 + pos_cam2 # + pos_correction_matrix2
        
        # print("ball real pos:" real_ball_pos)

        # print("ball pos error at cam 1: ", real_ball_pos - ball_pos_real_world1)
        # print("ball pos error at cam 2: ", real_ball_pos - ball_pos_real_world2)

        # cam1_error = cam1_error + (real_ball_pos - ball_pos_real_world1)
        # cam2_error = cam2_error + (real_ball_pos - ball_pos_real_world2)
       
        # ball_pos = (ball_pos_real_world1 + ball_pos_real_world2) / 2

        # print("pos error: ", real_ball_pos - ball_pos)

        ball_pos_que.append(ball_pos)
        if len(ball_pos_que) > 30:
            ball_pos_que.pop(0)

        try:
            prev_state = ball_pos_que[-2]
            curr_state = ball_pos_que[-1]
            vel = (curr_state - prev_state) / kalman_filter.dt
            vx = vel[0]
            vy = vel[1]
            vz = vel[2]
        except:
            vx=0;vy=0;vz=0


        # x_obs_list = [x[3] for x in state_filtered_que if x is not None]
        # y_obs_list = [x[3] for x in state_filtered_que if x is not None]
        # for x_obs, y_obs in zip(x_obs_list, y_obs_list):
        #     _, _ = kalman_filter.predict(x_obs, y_obs)

        # x_pred_list, y_pred_list, cov_pred_list = kalman_filter.predict(state_filtered[3], state_filtered[4], state_filtered[5], 30)
        # x_pred_list, y_pred_list, z_pred_list, cov_pred_list = kalman_filter.predict(ball_pos[0], ball_pos[1], ball_pos[2], vx, vy, vz, 1)

        # u_pred_list = y_pred_list
        # v_pred_list = z_pred_list
        # w_pred_list = x_pred_list


        # Done
        # cv2.circle(img_rgb1, (int(state_filtered[0]), int(state_filtered[1])), int(state_filtered[2]), (0, 255, 255), 2)
        # cv2.circle(img_rgb1, (int(state_filtered[3]), int(state_filtered[4])), 4, (0, 0, 255), -1)
        # for x_pred, y_pred, cov_pred in zip(x_pred_list, y_pred_list, cov_pred_list):
        #     # cv2.circle(img_rgb1, (int(x_pred), int(y_pred)), 5, (255, 0, 0), -1)
        #     cv2.circle(img_rgb1, (int(cam1._fx * x_pred), int(cam1._fx * y_pred)), int(np.sqrt(cov_pred[0, 0])), (255, 0, 0), 2)
        
        #노란색 코드드
        # cv2.circle(img_rgb1, (int(state_filtered1[0]), int(state_filtered1[1])), int(state_filtered1[3]), (0, 255, 255), 2)
        # cv2.circle(img_rgb2, (int(state_filtered2[0]), int(state_filtered2[1])), int(state_filtered2[3]), (0, 255, 255), 2)
        
        #파란색 코드
        # for u_pred, v_pred, w_pred, cov_pred in zip(u_pred_list, v_pred_list, w_pred_list, cov_pred_list):
        #     # cv2.circle(img_rgb1, (int(x_pred), int(y_pred)), 5, (255, 0, 0), -1)
        #     cv2.circle(img_rgb1, (int(cam1._fx*u_pred), int(-cam1._fx*v_pred)), int(np.sqrt(cov_pred[0, 0])), (255, 0, 0), 2)

        #빨간색 코드
        for j in range(1, len(ball_pos_que)):
            if (
                ball_pos_que[j - 1] is None or ball_pos_que[j] is None or
                not isinstance(ball_pos_que[j - 1], (list, tuple, np.ndarray)) or
                not isinstance(ball_pos_que[j], (list, tuple, np.ndarray)) or
                len(ball_pos_que[j - 1]) < 2 or len(ball_pos_que[j]) < 2
            ):
                continue

            pt1 = [int(x) for x in ball_pos_que[j - 1][0:2]]
            pt2 = [int(x) for x in ball_pos_que[j][0:2]]
            cv2.line(img_rgb1, pt1, pt2, (0, 0, 255), int(2 + 5 / float(len(ball_pos_que) - j)))

        # cv2.putText(img_rgb1, '{0:.3f}'.format(z_filtered), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2,
        #             (255, 255, 255), 10, cv2.LINE_AA)
        # cv2.putText(img_rgb1, '{0:.3f}'.format(z_filtered), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2,
        #             (0, 0, 0), 5, cv2.LINE_AA)

        # cv2.putText(img_rgb1, '{0:.3f}'.format(-state_filtered[4]), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 2,
        #            (255, 255, 255), 10, cv2.LINE_AA)
        # cv2.putText(img_rgb1, '{0:.3f}'.format(-state_filtered[4]), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 2,
        #            (0, 0, 0), 5, cv2.LINE_AA)

        tf = time.time_ns()

        if (tf - ts) / 1e9 > 0.01:
            cv2.putText(img_rgb1, '{0:02.1f} FPS'.format(1 / ((tf - ts) / 1e9)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 255, 255), 10, cv2.LINE_AA)
            cv2.putText(img_rgb1, '{0:02.1f} FPS'.format(1 / ((tf - ts) / 1e9)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 0), 5, cv2.LINE_AA)

        output1 = np.vstack((img_rgb1, img_depth1))
        output1 = cv2.resize(output1, dsize=(640, 720), interpolation=cv2.INTER_AREA)
        output2 = np.vstack((img_rgb2, img_depth2))
        output2 = cv2.resize(output2, dsize=(640, 720), interpolation=cv2.INTER_AREA)
        combined_output=np.hstack((output1, output2))
        cv2.imshow('RealSense', combined_output)

        # count = count + 1

        # Keyboard input
        k = cv2.waitKey(1) & 0xFF

        if k == 27:  # ESC
            cv2.destroyAllWindows()
            
            # print("error correction for cam1: ", cam1_error / count)
            # print("error correction: cam2: ", cam2_error / count)
            
            break#