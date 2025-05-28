# reference
# https://pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import time

from utils.Toolbox import *
from utils.Camera.realsense import RealSense, D455_DEFAULT_COLOR, D455_DEFAULT_DEPTH, L515_DEFAULT_DEPTH, L515_DEFAULT_COLOR

REALSENSE_SERIAL = "138322252637"
# REALSENSE_SERIAL = "f1371347"

def nothing(x):
    pass

class KalmanFilter:
    def __init__(self, num_memory=1):

        self.M = num_memory
        self.dt = np.float32(1/100)

        self.kf = cv2.KalmanFilter(6*self.M, 6*self.M, 1)
        # self.kf.measurementMatrix = np.zeros([3 * self.M, 6 * self.M], dtype=np.float32)
        self.kf.measurementMatrix = np.eye(6 * self.M, dtype=np.float32)
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

        self.kf.errorCovPre = np.identity(6*self.M, np.float32) * 1
        self.kf.measurementNoiseCov = np.identity(6*self.M, dtype=np.float32) * 3

        self.input = np.array([[np.float32(1)]])


    # def correct(self, x, y):
    #     measured = np.array([[np.float32(x)], [np.float32(y)]])
    #     self.kf.correct(measured)
    #     predicted = self.kf.predict(1).reshape(-1)
    #     return predicted

    def predict(self, x, y, z, vx, vy, vz, predict_interval):
        measured = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)], [np.float32(vx)], [np.float32(vy)], [np.float32(vz)]])
        statePost = self.kf.correct(measured)
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

    cam = RealSense(serial=REALSENSE_SERIAL)
    cam.initialize(resolution_color=D455_DEFAULT_COLOR, resolution_depth=D455_DEFAULT_DEPTH)

    ballLower = (6, 83, 149)
    ballUpper = (20, 237, 255)

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

    state_filtered = None # [x, y, r, xc, xy]
    z_filtered = None
    beta = 0.80
    ball_diameter = 0.04 # 3cm ball
    state_filtered_que = [None] * 30

    num_memory = 1 # need to implement?
    kalman_filter = KalmanFilter(num_memory)

    print(cam._fx, cam._fy)
    while True:
        ts = time.time_ns()

        # img_rgb, map_depth, img_depth = cam.get_color_depth('cv', clipping_depth=2.0)
        img_rgb, map_depth, img_depth = cam.get_color_depth('cv')
        # img_rgb = cv2.resize(img_rgb, dsize=(320, 180), interpolation=cv2.INTER_AREA)

        img_blur = cv2.GaussianBlur(img_rgb, (11, 11), 0)
        img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(img_hsv, ballLower, ballUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid

            # u,v,w are coordinates in image frame and x,y,z are coordinates in real world. related by scalar multiplication cam._fx and axis transformation
            # cam._fx = 628.456298828125, cam._fy = 627.6700439453125
            # u // y
            # v // -z
            # w // -x
            # x // -w
            # y // u
            # z // -v
            
            c = max(cnts, key=cv2.contourArea)

            ((u, v), r) = cv2.minEnclosingCircle(c)

            x = u / cam._fx
            y = v / cam._fx

            M = cv2.moments(c)
            uc = M["m10"] / M["m00"]
            vc = M["m01"] / M["m00"]
            wc = ball_diameter / (r * 2 / cam._fx)

            xc = - wc
            yc = uc / cam._fx
            zc = - vc / cam._fx

            state_observed = np.array([uc, vc, wc, r, xc, yc, zc])

            # only proceed if the radius meets a minimum size
            if r > 8:
                if state_filtered is None:
                    state_filtered = state_observed.copy()

                state_filtered = beta * state_observed + (1-beta) * state_filtered
            else:
                state_filtered = state_observed
        else:
            if state_filtered is None:
                state_filtered = np.zeros(7)

        z_filtered = map_depth[int(state_filtered[1]), int(state_filtered[0])]
        if z_filtered > 0:
            z_filtered = z_filtered + ball_diameter/2
        else:
            z_filtered = np.inf

        # z_estimated = cam._fx * ball_diameter / (state_filtered[2] * 2)

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
        cv2.circle(img_rgb, (int(state_filtered[0]), int(state_filtered[1])), int(state_filtered[2]), (0, 255, 255), 2)
        cv2.circle(img_rgb, (int(state_filtered[3]), int(state_filtered[4])), 4, (0, 0, 255), -1)

        # for x_pred, y_pred, cov_pred in zip(x_pred_list, y_pred_list, cov_pred_list):
        #     # cv2.circle(img_rgb, (int(x_pred), int(y_pred)), 5, (255, 0, 0), -1)
        #     cv2.circle(img_rgb, (int(cam._fx * x_pred), int(cam._fx * y_pred)), int(np.sqrt(cov_pred[0, 0])), (255, 0, 0), 2)

        for u_pred, v_pred, w_pred, cov_pred in zip(u_pred_list, v_pred_list, w_pred_list, cov_pred_list):
            # cv2.circle(img_rgb, (int(x_pred), int(y_pred)), 5, (255, 0, 0), -1)
            cv2.circle(img_rgb, (int(cam._fx*u_pred), int(-cam._fx*v_pred)), int(np.sqrt(cov_pred[0, 0])), (255, 0, 0), 2)

        for j in range(1, len(state_filtered_que)):
            if state_filtered_que[j - 1] is None or state_filtered_que[j] is None:
                continue
            pt1 = [int(x) for x in state_filtered_que[j - 1][0:2]]
            pt2 = [int(x) for x in state_filtered_que[j][0:2]]
            cv2.line(img_rgb, pt1, pt2, (0, 0, 255), int(2 + 5 / float(len(state_filtered_que) - j)))

        # cv2.putText(img_rgb, '{0:.3f}'.format(z_filtered), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2,
        #             (255, 255, 255), 10, cv2.LINE_AA)
        # cv2.putText(img_rgb, '{0:.3f}'.format(z_filtered), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2,
        #             (0, 0, 0), 5, cv2.LINE_AA)

        cv2.putText(img_rgb, '{0:.3f}'.format(-state_filtered[4]), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 255, 255), 10, cv2.LINE_AA)
        cv2.putText(img_rgb, '{0:.3f}'.format(-state_filtered[4]), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 0), 5, cv2.LINE_AA)

        cv2.circle(img_depth, (int(state_filtered[0]), int(state_filtered[1])), int(state_filtered[2]), (0, 255, 255), 2)
        cv2.circle(img_depth, (int(state_filtered[3]), int(state_filtered[4])), 4, (0, 0, 255), -1)

        tf = time.time_ns()

        if (tf-ts)/1e9 > 0.01:
            cv2.putText(img_rgb, '{0:02.1f} FPS'.format(1/((tf-ts)/1e9)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 255, 255), 10, cv2.LINE_AA)
            cv2.putText(img_rgb, '{0:02.1f} FPS'.format(1/((tf-ts)/1e9)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 0), 5, cv2.LINE_AA)

        output = np.vstack((img_rgb, img_depth))
        output = cv2.resize(output, dsize=(640, 720), interpolation=cv2.INTER_AREA)

        cv2.imshow('RealSense', output)

        # Keyboard input
        k = cv2.waitKey(1) & 0xFF

        if k == 27:  # ESC
            cv2.destroyAllWindows()
            break