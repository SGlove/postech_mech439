# reference
# https://pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import time
import pybullet as p
import pybullet_data

from utils.Toolbox import *
from utils.Camera.realsense import RealSense, D455_DEFAULT_COLOR, D455_DEFAULT_DEPTH, L515_DEFAULT_DEPTH, L515_DEFAULT_COLOR

REALSENSE_SERIAL_1 = "138322252637"
REALSENSE_SERIAL_2 = "138322250508"

def nothing(x):
    pass

# transfer XYZ euler angle to rotation matrix
def euler_2_rot_mat(phi, theta, psi):
    # define sin & cos for each euler angle
    cx, sx = np.cos(phi), np.sin(phi)
    cz, sz = np.cos(psi), np.sin(psi)
    cy, sy = np.cos(theta), np.sin(theta)
    
    # define rotation matrix for each xyz axis
    Rx = np.array([[  1,   0,   0], 
                   [  0,  cx, -sx],
                   [  0,  sx,  cx]])
    Ry = np.array([[ cy,   0, sy],
                   [  0,   1,  0],
                   [-sy,   0, cy]])
    Rz = np.array([[ cz, -sz, 0],
                   [ sz,  cz, 0],
                   [  0,   0, 1]])
    
    return Rx @ Ry @ Rz

def cam_to_world(p_cam, cam_pos, cam_euler):
    R = euler_2_rot_mat(*cam_euler)
    return R.dot(p_cam) + cam_pos

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
    # 두 대의 카메라 초기화
    cam1 = RealSense(serial=REALSENSE_SERIAL_1)
    cam1.initialize(resolution_color=D455_DEFAULT_COLOR, resolution_depth=D455_DEFAULT_DEPTH)

    cam2 = RealSense(serial=REALSENSE_SERIAL_2)
    cam2.initialize(resolution_color=D455_DEFAULT_COLOR, resolution_depth=D455_DEFAULT_DEPTH)

    # global coordinate system에서 각 camera의 위치(x,y,z) 및 euler angle 설정
    x1, y1, z1         = 0.15, 0.00, 0.15
    theta1, phi1, psi1 = 0.00, 0.00, 0.00

    x2, y2, z2         = 0.00, 0.15, 0.15
    theta2, phi2, psi2 = 0.00, 0.00, 0.00

    # 카메라별 월드 프레임에서의 위치 및 회전 (변수로 지정)
    c1_pos   = np.array([x1, y1, z1])            # cam1 위치 (m)
    c1_euler = np.array([theta1, phi1, psi1])    # cam1 (yaw, pitch, roll) in rad
    c2_pos   = np.array([x2, y2, z2])            # cam2 위치 (m)
    c2_euler = np.array([theta2, phi2, psi2])    # cam2 (yaw, pitch, roll) in rad

    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,        0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,      0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0,0,-9.81)
    plane = p.loadURDF("plane.urdf")

    # 구체(공) 생성: 반지름 0.02m, 빨간색
    sphere_vis = p.createVisualShape(
        shapeType =p.GEOM_SPHERE,
        radius = 0.1,
        rgbaColor = [1,1,1,1]
    )

    sphere = p.createMultiBody(
        baseMass=0,                     # 물리 시뮬 필요 없으므로 질량 0
        baseVisualShapeIndex=sphere_vis,
        basePosition=[0,0,0]
    )

    text_id = p.addUserDebugText(
    text="",
    textPosition=[0,0,0],        # 나중에 덮어씌워질 좌표
    textColorRGB=[1,1,0],        # 노란색
    textSize=1.2
    )

    # ───── 카메라 시각화 ───────────────────────────
    # 작고 얇은 박스로 카메라 위치 표시
    cam_size = [0.05, 0.02, 0.01]  # x,y,z 크기
    cam_color = [0,1,0,1]
    cam_box_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=cam_size,
        rgbaColor=cam_color
    )
    cam1_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=cam_box_vis,
                                  basePosition=c1_pos.tolist(),
                                  baseOrientation=p.getQuaternionFromEuler(c1_euler.tolist()))
    cam2_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=cam_box_vis,
                                  basePosition=c2_pos.tolist(),
                                  baseOrientation=p.getQuaternionFromEuler(c2_euler.tolist()))

    ballLower = (6, 83, 149)
    ballUpper = (20, 237, 255)
    beta = 0.80
    ball_diameter = 0.04  # 4cm

    cv2.namedWindow('Cam1', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Cam2', cv2.WINDOW_AUTOSIZE)

    state_f1 = None
    state_f2 = None
    que1 = [None]*30
    que2 = [None]*30

    kalman_filter = KalmanFilter(num_memory=1)

    print("Cam1 fx,fy:", cam1._fx, cam1._fy)
    print("Cam2 fx,fy:", cam2._fx, cam2._fy)

    while True:
        ts = time.time_ns()

        # 3) 프레임 획득
        img1, map1, depth1 = cam1.get_color_depth('cv')
        img2, map2, depth2 = cam2.get_color_depth('cv')

        # 4) 공 탐지 + 3D 좌표 계산 함수
        def detect_and_filter(img, depth_map, state_f, que, cam):

            xc = yc = zc = 0.0
            u = v = r = 0.0
            
            blur = cv2.GaussianBlur(img, (11,11), 0)
            hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, ballLower, ballUpper)
            mask = cv2.erode(mask, None, 2)
            mask = cv2.dilate(mask, None, 2)
            cnts = imutils.grab_contours(
                cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            )

            if cnts:
                c = max(cnts, key=cv2.contourArea)
                (u,v), r = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                uc = M["m10"]/M["m00"] if M["m00"] else u
                vc = M["m01"]/M["m00"] if M["m00"] else v
                wc = ball_diameter/(r*2/cam._fx)

                # 카메라 로컬 3D 좌표
                xc = -wc
                yc = uc/cam._fx
                zc = -vc/cam._fx

                obs = np.array([uc, vc, wc, r, xc, yc, zc])
                if r>8:
                    state_f = obs if state_f is None else beta*obs + (1-beta)*state_f
                else:
                    state_f = obs
            else:
                if state_f is None:
                    state_f = np.zeros(7)

            # 깊이 보정
            zf = depth_map[int(state_f[1]), int(state_f[0])]
            zf = (zf + ball_diameter/2) if zf>0 else np.inf

            # 속도 계산
            que.append(state_f)
            if len(que)>30: que.pop(0)
            try:
                prev, curr = que[-2], que[-1]
                vel = (curr-prev)/kalman_filter.dt
                vx, vy, vz = vel[4], vel[5], vel[6]
            except:
                vx=vy=vz=0

            # Kalman 예측 (1 step)
            xpl, ypl, zpl, covl = kalman_filter.predict(
                state_f[4], state_f[5], state_f[6], vx, vy, vz, 1
            )

            return state_f, (xc,yc,zc), (u,v,r), (xpl,ypl,zpl,covl)

        # 5) cam1, cam2 각각 처리
        state_f1, p1_cam, pix1, pred1 = detect_and_filter(img1, map1, state_f1, que1, cam1)
        state_f2, p2_cam, pix2, pred2 = detect_and_filter(img2, map2, state_f2, que2, cam2)

        # 6) 로컬→월드 변환
        p_w1 = cam_to_world(np.array(p1_cam), c1_pos, c1_euler)
        p_w2 = cam_to_world(np.array(p2_cam), c2_pos, c2_euler)

        # 7) 두 결과 평균 (월드 좌표)
        p_world = (p_w1 + p_w2)/2

        p.resetBasePositionAndOrientation(
            sphere,
            posObj=list(p_world),
            ornObj=[0,0,0,1]
        )

        p.addUserDebugText(
        text=f"x={p_world[0]:.2f}, y={p_world[1]:.2f}, z={p_world[2]:.2f}",
        textPosition=(p_world + np.array([0,0,0.1])).tolist(),  
            # 공 바로 위(0.1m)
        textColorRGB=[1,1,0],
        textSize=1.2,
        replaceItemUniqueId=text_id
        )

        p.stepSimulation()

        # 8) 시각화: 공 중심과 예측 그리기
        u1,v1,r1 = pix1
        cv2.circle(img1, (int(u1),int(v1)), int(r1),(0,255,255),2)
        # 월드 좌표를 다시 픽셀화해서 표시 (cam1 관점)
        px1 = int((p_world[1]*cam1._fx));  py1 = int((-p_world[2]*cam1._fx))
        cv2.circle(img1, (px1,py1), 5, (0,0,255), -1)
        cv2.putText(img1, f"W: {p_world.round(2)}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)

        u2,v2,r2 = pix2
        cv2.circle(img2, (int(u2),int(v2)), int(r2),(0,255,255),2)
        # cam2 관점 표시
        px2 = int((p_world[1]*cam2._fx));  py2 = int((-p_world[2]*cam2._fx))
        cv2.circle(img2, (px2,py2), 5, (0,0,255), -1)

        cv2.imshow('Cam1', img1)
        cv2.imshow('Cam2', img2)

        if cv2.waitKey(1)&0xFF == 27:
            break
    
    p.disconnect()
    cv2.destroyAllWindows()