import pinocchio as pin
import numpy as np
import math

from .robotics_utils import *

def getLinearPath(T_start, T_end, N=500):

    Tlist = np.zeros([N, 4, 4])

    R_start = T_start[0:3, 0:3]
    p_start = T_start[0:3, 3:4]
    R_end = T_end[0:3, 0:3]
    p_end = T_end[0:3, 3:4]

    xi_ceil = MatrixLog3(R_start.T @ R_end)

    TimeScaling = getTimeScaling(N, POLYNOMIAL3)

    for i in range(N):
        Tlist[i, 0:3, 0:3] = R_start @ MatrixExp3(xi_ceil * TimeScaling[i, 0])
        Tlist[i, 0:3, 3:4] = p_start + (p_end - p_start) * TimeScaling[i, 0]
        Tlist[i, 3, 3] = 1

    return Tlist

def getScrewPath(T_start, T_end, N=500):

    Tlist = np.zeros([N, 4, 4])

    lamb_ceil = MatrixLog6(TransInv(T_start) @ T_end)

    TimeScaling = getTimeScaling(N, POLYNOMIAL3)

    for i in range(N):
        Tlist[i, :, :] = T_start @ MatrixExp6(lamb_ceil * TimeScaling[i, 0])

    return Tlist


def Tlist2Jointlist(Tlist, q_start, pinModel):

    assert type(q_start) is list

    N = np.shape(Tlist)[0]
    numJoint=len(q_start)

    qlist = np.zeros([numJoint, N])
    qlist[:, 0] = np.asarray(q_start)

    damp = 1e-10

    for i in range(N-1):
        T_curr = pinModel.FK(qlist[:,[i]])
        J = pinModel.Js(qlist[:,[i]])

        # Jinv = J.T@np.linalg.pinv(J@J.T + damp * np.eye(6))
        # J1_temp = np.identity(numJoint) - Jinv @ J
        V = se3ToVec((Tlist[i + 1, :, :] - T_curr) @ TransInv(Tlist[i + 1, :, :]))
        dp_dq = np.power(qlist[:, [i]], 3)

        null_1 = J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), V))
        null_2 = (np.identity(numJoint) - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), J))) @ dp_dq

        # qlist[:,[i+1]] = qlist[:,[i]] + (Jinv @ V -0.001* J1_temp @ np.power(qlist[:,[i]], 3))
        qlist[:, [i+1]] = qlist[:, [i]] + (null_1 - 0.001 * null_2)

    return qlist

POLYNOMIAL3 = 0
POLYNOMIAL5 = 1
TRAPEZOIDAL = 2
SCURVE = 3

C_SPLINE_ZEROVEL = 0
C_SPLINE_NONZEROVEL = 1

B_SPLINE = 2


class WayPoint():
    def __init__(self, time, pos, R, q):
        self.time = time
        self.pos = pos
        self.R = R
        self.q = q


def getTimeScaling(total_itr=1000, v=0, a=0, dt=1./240., TimeScalingType=POLYNOMIAL3):


    TimeScaling = np.zeros([total_itr, 1])

    for i in range(total_itr):
        if (TimeScalingType == POLYNOMIAL3):
            TimeScaling[i, 0] = 3 * ((i + 1) / total_itr) ** 2 - 2 * ((i + 1) / total_itr) ** 3
        elif(TimeScalingType == POLYNOMIAL5):
            TimeScaling[i, 0] = 10 * ((i + 1) / total_itr) ** 3 - 15 * ((i + 1) / total_itr) ** 4 + 6 * ((i + 1) / total_itr) ** 5



    return TimeScaling


def getdTimeScaling(total_itr, TimeScalingType=POLYNOMIAL3):

    dTimeScaling = np.zeros([total_itr, 1])

    for i in range(total_itr):
        if (TimeScalingType == POLYNOMIAL3):
            dTimeScaling[i, 0] = 6/total_itr * (((i + 1) / total_itr) - ((i + 1) / total_itr) ** 2)
        elif(TimeScalingType == POLYNOMIAL5):
            dTimeScaling[i, 0] = 30/total_itr(((i + 1) / total_itr) ** 2 - 2 * ((i + 1) / total_itr) ** 3 + ((i + 1) / total_itr) ** 4)
        # elif(TimeScalingType == SCURVE):
        #     if(v_max**2/a_max > 1):
        #         print("INPUT ERROR!!! TRY AGAIN!!!")
        #         exit()
        #     else:

    return dTimeScaling


def calculate_C_Spline(Waypoint_list, mode):
    '''
    IndyCustomControl_visualization.py에서 CalculateTrajectory()함수 사용할 때 SplineType에 따라 mode가 달라지도록 설정함.
    pybullet.CalculateTrajectory(0) : C-Spline with zero velocity
    pybullet.CalculateTrajectory(1) : C-Spline with nonzero velocity
    pybullet.CalculateTrajectory(2) : B-Spline
    '''

    numWaypoint = np.shape(Waypoint_list)[0]
    print("shape={}".format(numWaypoint))
    _robot_freq = 2000  ## 추후 로봇에서 받아오든지 수정하든지 해야함

    if (numWaypoint != 0 and numWaypoint != 1):

        numTrajectorypoint = int((Waypoint_list[-1].time - Waypoint_list[0].time)*_robot_freq)
        pos_list = np.zeros([3, numTrajectorypoint])
        rot_list = np.zeros([numTrajectorypoint, 3, 3])

        _cnt = 0

        for i in range(numWaypoint-1):
            period_segment = Waypoint_list[i+1].time - Waypoint_list[i].time
            num_interpolated = int(period_segment * _robot_freq)

            if (mode == 1):
                if (i==0):
                    pdot_1 = np.zeros([3, 1])
                else:
                    pdot_1 = (Waypoint_list[i+1].pos - Waypoint_list[i].pos)/period_segment

                if (i == numWaypoint - 2):
                    pdot_2 = np.zeros([3, 1])
                else:
                    pdot_2 = (Waypoint_list[i+2].pos - Waypoint_list[i+1].pos)/period_segment
            else:
                pdot_1 = np.zeros([3, 1])
                pdot_2 = np.zeros([3, 1])

            a0 = Waypoint_list[i].pos
            a1 = pdot_1
            a2 = (3*(Waypoint_list[i+1].pos-Waypoint_list[i].pos)-(2*pdot_1+pdot_2)*period_segment)/(period_segment**2)
            a3 = (2*(Waypoint_list[i].pos-Waypoint_list[i+1].pos)+(pdot_1+pdot_2)*period_segment)/(period_segment**3)

            s = getTimeScaling(num_interpolated, TimeScalingType=POLYNOMIAL3)

            for k in range(num_interpolated):
                dt_ = period_segment * (k+1)/num_interpolated
                pos_list[:, [_cnt]] = a0 + a1*dt_ + a2*dt_**2 + a3*dt_**3

                rot_list[[_cnt], :, :] \
                    = Waypoint_list[i].R @ MatrixExp3(MatrixLog3((Waypoint_list[i].R).T @ Waypoint_list[i+1].R)*s[k, 0])

                _cnt += 1

        return [True, pos_list, rot_list]
    else:
        print('\033[1m' + '\033[91m' + "There are no sufficient waypoints!!!" + '\033[0m\n')
        print('\033[91m' + "shape={}".format(numWaypoint) + '\033[0m\n')

        return [False, None, None]


def calculate_B_Spline(Waypoint_list, B_Spline_ratio):

    numWaypoint = np.shape(Waypoint_list)[0]
    print("shape={}".format(numWaypoint))
    _robot_freq = 2000  ## 추후 로봇에서 받아오든지 수정하든지 해야함

    d1_list = np.zeros([numWaypoint - 1, 1])
    d2_list = np.zeros([numWaypoint - 1, 1])
    v_list = np.zeros([numWaypoint - 1, 1])
    K_list = np.zeros([3, numWaypoint - 1])

    time_betway = np.zeros([numWaypoint - 1, 1]);

    for i in range(numWaypoint-1):
        time_betway[i, 0] = Waypoint_list[i + 1].time - Waypoint_list[i].time
    if (B_Spline_ratio > 1):
        print("Bspline ratio value is too large! The value was lowered to 0.5")
        B_Spline_ratio = 0.5

    min_waytime = np.min(time_betway)
    delta_T = min_waytime * B_Spline_ratio

    #delta_T = 2

    if (numWaypoint != 0 and numWaypoint != 1):

        '''
        먼저, d1과 d2를 계산합니다.
        '''
        for i in range(numWaypoint-1):
            period_segment = Waypoint_list[i + 1].time - Waypoint_list[i].time # i번째 way를 이동하는 총 시간

            v = (Waypoint_list[i+1].pos - Waypoint_list[i].pos)/period_segment # i번째 way를 이동하는 속도
            v_list[i, 0] = np.linalg.norm(v)
            K_list[:, [i]] = v/v_list[i, 0]

            if (i != numWaypoint-2):
                d1_list[i, 0] = v_list[i, 0] * delta_T / 2
            if (i != 0):
                d2_list[i, 0] = d1_list[i - 1, 0] * v_list[i, 0] / v_list[i - 1, 0]

        '''
        그 후, 계산된 d1과 d2를 기반으로 B-Spline을 계산합니다. 
        '''
        numTrajectorypoint = (Waypoint_list[-1].time - Waypoint_list[0].time) * _robot_freq
        print("tp = {}".format(numTrajectorypoint))
        pos_list = np.zeros([3, numTrajectorypoint])
        rot_list = np.zeros([numTrajectorypoint, 3, 3])

        _cnt = 0
        for i in range(numWaypoint - 1):

            if numWaypoint == 2:
                period_segment = Waypoint_list[i + 1].time - Waypoint_list[i].time
            elif i == 0:
                period_segment = Waypoint_list[i + 1].time - Waypoint_list[i].time + delta_T / 2
            elif i == numWaypoint-2:
                period_segment = Waypoint_list[i + 1].time - Waypoint_list[i].time - delta_T / 2
            else:
                period_segment = Waypoint_list[i + 1].time - Waypoint_list[i].time

            num_interpolated = int(period_segment * _robot_freq)

            p1 = Waypoint_list[i].pos + d2_list[i, 0] * K_list[:, [i]]

            s = getTimeScaling(num_interpolated, TimeScalingType=POLYNOMIAL3)

            for k in range(num_interpolated):
                dt_ = period_segment * (k + 1) / num_interpolated
                dt_a = period_segment * (num_interpolated - delta_T*_robot_freq + 1) / num_interpolated # 곡선으로 움직이기 시작하는 시간

                if (k <= num_interpolated - delta_T*_robot_freq or i == numWaypoint-2):
                    pos_list[:, [_cnt]] = p1 + v_list[i, 0]*K_list[:, [i]]*dt_
                else:
                    pos_list[:, [_cnt]] = p1 + v_list[i, 0]*K_list[:, [i]]*dt_a + v_list[i, 0]*K_list[:, [i]]*(dt_ - dt_a)\
                    + (v_list[i+1, 0]*K_list[:, [i+1]] - v_list[i, 0]*K_list[:, [i]])*(dt_-dt_a)**2/(2*delta_T)

                    # p1 + v_list[i, 0]*K_list[:, [i]]*dt_a = A'

                rot_list[[_cnt], :, :] \
                    = Waypoint_list[i].R @ MatrixExp3(
                    MatrixLog3((Waypoint_list[i].R).T @ Waypoint_list[i + 1].R) * s[k, 0])

                _cnt += 1

        return [True, pos_list, rot_list]

    else:
        print('\033[1m' + '\033[91m' + "There are no sufficient waypoints!!!" + '\033[0m\n')
        print('\033[91m' + "shape={}".format(numWaypoint) + '\033[0m\n')

        return [False, None, None]