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

from utils.Toolbox import *
from utils.Camera.realsense import RealSense, D455_DEFAULT_COLOR, D455_DEFAULT_DEPTH, L515_DEFAULT_DEPTH, L515_DEFAULT_COLOR

import threading
import keyboard



########################### vision code ###########################


SERIAL1 = "138322252637"
SERIAL2 = "138322250508"

transform_matrix = np.eye(4)
ball_pos = [0, 0, 0]
ball_vel = [0, 0, 0]
is_cam_setup = False
stop_camera = False

def nothing(x):
    pass

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

def runCamera():
    global ball_pos
    global ball_vel
    global is_cam_setup

    cam1 = RealSense(serial=SERIAL1)
    cam1.initialize(resolution_color=D455_DEFAULT_COLOR, resolution_depth=D455_DEFAULT_DEPTH)

    cam2 = RealSense(serial=SERIAL2)
    cam2.initialize(resolution_color=D455_DEFAULT_COLOR, resolution_depth=D455_DEFAULT_DEPTH)

    ballLower = (6, 126, 219)
    ballUpper = (78, 255, 255)

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE) #UI

    state_filtered = None # [x, y, r, xc, xy]
    state_filtered1 = None
    state_filtered2 = None
    z_filtered = None
    beta = 0.80
    ball_diameter = 0.05 # 4cm ball
    state_filtered_que = [None] * 30

    num_memory = 1 # need to implement?
    kalman_filter = KalmanFilter(num_memory)

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

            x1 = u1 / cam1._fx
            y1 = v1 / cam1._fx

            M1 = cv2.moments(c1)
            uc1 = M1["m10"] / M1["m00"]
            vc1 = M1["m01"] / M1["m00"]
            wc1 = ball_diameter / (r1 * 2 / cam1._fx)

            xc1 = - wc1
            yc1 = uc1 / cam1._fx
            zc1 = - vc1 / cam1._fx

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

            x2 = u2 / cam2._fx
            y2 = v2 / cam2._fx

            M2 = cv2.moments(c2)
            uc2 = M2["m10"] / M2["m00"]
            vc2 = M2["m01"] / M2["m00"]
            wc2 = ball_diameter / (r2* 2 / cam2._fx)

            xc2 = - wc2
            yc2 = uc2 / cam2._fx
            zc2 = - vc2 / cam2._fx

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

        #임시시
        # state_filtered=state_filtered1
        # print(state_filtered1)
        # print("1")
        # print(state_filtered)
        # print(state_filtered2)
        # z_estimated = cam1._fx * ball_diameter / (state_filtered[2] * 2)

        ball_pos[0] = state_filtered[0]
        ball_pos[1] = state_filtered[2]
        ball_pos[2] = state_filtered[1]

        state_filtered_que.append(state_filtered)
        if len(state_filtered_que) > 30:
            state_filtered_que.pop(0)

        try:
            prev_state = state_filtered_que[-2]
            curr_state = state_filtered_que[-1]
            vel = (curr_state - prev_state) / kalman_filter.dt
            vx = vel[4]
            vy = vel[5]
            vz = vel[6]
        except:
            vx=0;vy=0;vz=0

        ball_vel[0] = vx
        ball_vel[1] = vy
        ball_vel[2] = vz

        # x_obs_list = [x[3] for x in state_filtered_que if x is not None]
        # y_obs_list = [x[3] for x in state_filtered_que if x is not None]
        # for x_obs, y_obs in zip(x_obs_list, y_obs_list):
        #     _, _ = kalman_filter.predict(x_obs, y_obs)

        # x_pred_list, y_pred_list, cov_pred_list = kalman_filter.predict(state_filtered[3], state_filtered[4], state_filtered[5], 30)
        x_pred_list, y_pred_list, z_pred_list, cov_pred_list = kalman_filter.predict(state_filtered[0], state_filtered[1], state_filtered[2], vx, vy, vz, 1)

        u_pred_list = y_pred_list
        v_pred_list = z_pred_list
        w_pred_list = x_pred_list


        # Done
        # cv2.circle(img_rgb1, (int(state_filtered[0]), int(state_filtered[1])), int(state_filtered[2]), (0, 255, 255), 2)
        # cv2.circle(img_rgb1, (int(state_filtered[3]), int(state_filtered[4])), 4, (0, 0, 255), -1)
        # for x_pred, y_pred, cov_pred in zip(x_pred_list, y_pred_list, cov_pred_list):
        #     # cv2.circle(img_rgb1, (int(x_pred), int(y_pred)), 5, (255, 0, 0), -1)
        #     cv2.circle(img_rgb1, (int(cam1._fx * x_pred), int(cam1._fx * y_pred)), int(np.sqrt(cov_pred[0, 0])), (255, 0, 0), 2)
        # UI
        #노란색 코드드        
        cv2.circle(img_rgb1, (int(state_filtered[0]), int(state_filtered[1])), int(state_filtered[3]), (0, 255, 255), 2)
        #파란색 코드
        for u_pred, v_pred, w_pred, cov_pred in zip(u_pred_list, v_pred_list, w_pred_list, cov_pred_list):
            # cv2.circle(img_rgb1, (int(x_pred), int(y_pred)), 5, (255, 0, 0), -1)
            cv2.circle(img_rgb1, (int(cam1._fx*u_pred), int(-cam1._fx*v_pred)), int(np.sqrt(cov_pred[0, 0])), (255, 0, 0), 2)

        #빨간색 코드
        for j in range(1, len(state_filtered_que)):
            if state_filtered_que[j - 1] is None or state_filtered_que[j] is None:
                continue
            pt1 = [int(x) for x in state_filtered_que[j - 1][0:2]]
            pt2 = [int(x) for x in state_filtered_que[j][0:2]]
            cv2.line(img_rgb1, pt1, pt2, (0, 0, 255), int(2 + 5 / float(len(state_filtered_que) - j)))

        # cv2.putText(img_rgb1, '{0:.3f}'.format(z_filtered), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2,
        #             (255, 255, 255), 10, cv2.LINE_AA)
        # cv2.putText(img_rgb1, '{0:.3f}'.format(z_filtered), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2,
        #             (0, 0, 0), 5, cv2.LINE_AA)

        cv2.putText(img_rgb1, '{0:.3f}'.format(-state_filtered[4]), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 255, 255), 10, cv2.LINE_AA)
        cv2.putText(img_rgb1, '{0:.3f}'.format(-state_filtered[4]), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 0), 5, cv2.LINE_AA)

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




indy = IndyDCP3(robot_ip='192.168.0.22', index=0)

def print_coordinate():
    print("x: " + str(ball_pos[0]) + " y: " + str(ball_pos[1]) + " z: " + str(ball_pos[2]))

def robot_calibration(home_pos : np.ndarray, w, h):
    print("#### starting calibration... ####")
    time_per_step = 2.5
    vel_ratio = 0.5
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
    time.sleep(time_per_step)
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


def compute_linear_roll_pitch(x, y, width, max_degree):
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

def bounce(current_robot_pos: np.ndarray, target_ball_pos: np.ndarray, bounce_force ):
    indy.movetelel_abs(tpos = current_robot_pos - bounce_force * np.array([0, 0, 10, 0, 0, 0]))
    time.sleep(0.05 * bounce_force)
    indy.movetelel_abs(tpos = current_robot_pos)
    time.sleep(0.05 * bounce_force)











########################### main code ###########################

indy.stop_teleop()

home_pos = np.array([570, 10, 380, -85, 75.5, -90]) # x, y, z (mm), x, y, z (deg)
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

app = QApplication(sys.argv)
ui = BallPositionUI()
ui.show()

try:
    workspace_width = 350
    workspace_height = 250
    workspace_tolerance = 50
    vel_threshold = 1.5
    robot_calibration(home_pos, workspace_width, workspace_height)
    time.sleep(2)
    indy.movetelel_abs(home_pos, 0.5, 1.0)
    time.sleep(2)

    target_z = -workspace_height/2
    bounce_height = 120

    vel_data = open("file_velocity_data.txt", 'w')
    time_data = open("file_time_data.txt", 'w')
    z_pos_data = open("file_z_pos_data.txt", 'w')

    start_time = time.time()
    count = 0

    print("#### detach the ball and press s ####")
    while (True):
        if (keyboard.is_pressed('s')):
            break
        time.sleep(0.1)

    while (True):
        if (keyboard.is_pressed('esc')):
            break
        
        now_time = time.time()
        vel_data.write(str(ball_vel[0]) + "," + str(ball_vel[1]) + "," + str(ball_vel[2]) + "\n")
        time_data.write(str(now_time) + "\n")

        count = count + 1
        if (now_time - start_time > 1):
            print("sampling per second : " + str(count))
            count = 0
            start_time = now_time

        ball_pos_ws = transform_point((ball_pos[0], ball_pos[1], ball_pos[2]))

        z_pos_data.write(str(ball_pos_ws[2]) + "\n")

        if ((np.abs(ball_pos_ws[0]) > (workspace_width/2 + workspace_tolerance))
            or (np.abs(ball_pos_ws[1]) > (workspace_width/2 + workspace_tolerance))):
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

        #print("vel_z: " + str(ball_vel[2]))

        if (ball_vel[2] > vel_threshold):
            target_z = -workspace_height/2
        elif (ball_vel[2] < -vel_threshold):
            target_z = -workspace_height/2 + bounce_height

        #indy.movetelel_abs(home_pos + np.array([0, 0, target_z, 0, 0, 0]), 1.0, 1.0)
        roll_rad, pitch_rad = compute_linear_roll_pitch(ball_pos_ws[0], ball_pos_ws[1], workspace_width, 10)
        lacket_angles = apply_roll_pitch(home_pos[3:6], roll_rad, pitch_rad)
        indy.movetelel_abs(np.array([home_pos[0] + ball_pos_ws[0], home_pos[1] + ball_pos_ws[1], home_pos[2] + target_z,
                                    lacket_angles[0], lacket_angles[1], lacket_angles[2]]))
        #indy.movetelel_abs(home_pos + np.array([ball_pos_ws[0], ball_pos_ws[1], -workspace_height/2, 0, 0, 0]))
    
except Exception as e:
    print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

finally:
    indy.stop_teleop()
    stop_camera = True
    vel_data.close()
    time_data.close()
    z_pos_data.close()
    sys.exit(app.exec_())
