# reference
# https://pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import time

from utils.Toolbox import *
from utils.Camera.realsense import RealSense, D455_DEFAULT_COLOR, D455_DEFAULT_DEPTH, L515_DEFAULT_DEPTH, L515_DEFAULT_COLOR

SERIAL1 = "138322252637"
SERIAL2 = "138322250508"

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

if __name__ == '__main__':

    cam1 = RealSense(serial=SERIAL1)
    cam1.initialize(resolution_color=D455_DEFAULT_COLOR, resolution_depth=D455_DEFAULT_DEPTH)

    cam2 = RealSense(serial=SERIAL2)
    cam2.initialize(resolution_color=D455_DEFAULT_COLOR, resolution_depth=D455_DEFAULT_DEPTH)

    ballLower = (6, 83, 149)
    ballUpper = (20, 237, 255)

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

    state_filtered = None # [x, y, r, xc, xy]
    state_filtered1 = None
    state_filtered2 = None
    z_filtered = None
    beta = 0.80
    ball_diameter = 0.05 # 4cm ball
    state_filtered_que = [None] * 30

    num_memory = 1 # need to implement?
    kalman_filter = KalmanFilter(num_memory)

    print(cam1._fx, cam1._fy)
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

            state_observed1 = np.array([uc1, vc1, wc1, r1, xc1, yc1, zc1])

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

            state_observed2 = np.array([uc2, vc2, wc2, r2, xc2, yc2, zc2])

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

        state_filtered=(state_filtered1+state_filtered2)/2

        z_filtered = map_depth1[int(state_filtered[1]), int(state_filtered[0])]
        if z_filtered > 0:
            z_filtered = z_filtered + ball_diameter/2
        else:
            z_filtered = np.inf

        # z_estimated = cam1._fx * ball_diameter / (state_filtered[2] * 2)

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

        # x_obs_list = [x[3] for x in state_filtered_que if x is not None]
        # y_obs_list = [x[3] for x in state_filtered_que if x is not None]
        # for x_obs, y_obs in zip(x_obs_list, y_obs_list):
        #     _, _ = kalman_filter.predict(x_obs, y_obs)

        # x_pred_list, y_pred_list, cov_pred_list = kalman_filter.predict(state_filtered[3], state_filtered[4], state_filtered[5], 30)
        x_pred_list, y_pred_list, z_pred_list, cov_pred_list = kalman_filter.predict(state_filtered[4], state_filtered[5], state_filtered[6], vx, vy, vz, 1)

        u_pred_list = y_pred_list
        v_pred_list = z_pred_list
        w_pred_list = x_pred_list


        # Done
        # cv2.circle(img_rgb1, (int(state_filtered[0]), int(state_filtered[1])), int(state_filtered[2]), (0, 255, 255), 2)
        # cv2.circle(img_rgb1, (int(state_filtered[3]), int(state_filtered[4])), 4, (0, 0, 255), -1)
        # for x_pred, y_pred, cov_pred in zip(x_pred_list, y_pred_list, cov_pred_list):
        #     # cv2.circle(img_rgb1, (int(x_pred), int(y_pred)), 5, (255, 0, 0), -1)
        #     cv2.circle(img_rgb1, (int(cam1._fx * x_pred), int(cam1._fx * y_pred)), int(np.sqrt(cov_pred[0, 0])), (255, 0, 0), 2)

        for u_pred, v_pred, w_pred, cov_pred in zip(u_pred_list, v_pred_list, w_pred_list, cov_pred_list):
            # cv2.circle(img_rgb1, (int(x_pred), int(y_pred)), 5, (255, 0, 0), -1)
            cv2.circle(img_rgb1, (int(cam1._fx*u_pred), int(-cam1._fx*v_pred)), int(np.sqrt(cov_pred[0, 0])), (255, 0, 0), 2)

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

        cv2.circle(img_depth1, (int(state_filtered[0]), int(state_filtered[1])), int(state_filtered[2]), (0, 255, 255), 2)
        cv2.circle(img_depth1, (int(state_filtered[3]), int(state_filtered[4])), 4, (0, 0, 255), -1)

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

        # Keyboard input
        k = cv2.waitKey(1) & 0xFF

        if k == 27:  # ESC
            cv2.destroyAllWindows()
            break