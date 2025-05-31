import numpy as np
from scipy.spatial.transform import Rotation as ro

def compute_paddle_orientation(ball_pos, ball_vel,
                               apex_height=0.25,   # in m it's height above the racket    
                               restitution=0.92,      # to cha,ge when we will test it (between 0.88–0.94)
                               damping=1.0,           # 1 = full lateral correction in one hit
                               g=9.81):


    x, y, _          = ball_pos
    vx_in, vy_in, vz_in = ball_vel

    # vertical speed needed for chosen apex
    vz_out = np.sqrt(2 * g * apex_height)

    # time until that apex, then horizontal speeds that cancel x-y error
    t_up   = vz_out / g
    vx_out = -damping * x / t_up
    vy_out = -damping * y / t_up
    v_des  = np.array([vx_out, vy_out, vz_out])

    v_in   = np.array([vx_in, vy_in, vz_in])
    n      = (v_in - v_des) / ((1 + restitution) *
                               np.linalg.norm(v_in - v_des))

    # normal points upward (nz > 0)
    if n[2] < 0:
        n = -n
    nx, ny, nz = n      

    roll  = -np.arcsin(ny)   # about +/-X (to test it to know)
    pitch = np.arcsin(nx)   # about +/-Y (same)

    return roll, pitch

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

home_ori = [95, 104.5, 90]
ball_pos = np.array([np.float64(1.13), np.float64(10.13), np.float64(30.13)])
ball_vel = np.array([np.float64(0.5), np.float64(0.5), np.float64(-3.2)])

roll, pitch = compute_paddle_orientation(ball_pos, ball_vel)
print("roll: ", np.degrees(roll), ", ptich: ", np.degrees(pitch))
new_angles = apply_roll_pitch(home_ori, roll, pitch)
print("New XYZ Euler angles (deg):", new_angles)
