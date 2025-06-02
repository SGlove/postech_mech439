import matplotlib.pyplot as plt

# 파일 로드 함수
def load_xyz(filename):
    x, y, z = [], [], []
    with open(filename) as f:
        for line in f:
            a, b, c = map(float, line.strip().split(','))
            x.append(a)
            y.append(b)
            z.append(c)
    return x, y, z

# 시간 데이터 로드 및 보정
with open('file_time_data.txt') as f:
    time_data = [float(line.strip()) for line in f]
base_time = time_data[0]
time_data = [t - base_time for t in time_data]

# 공 위치 / 로봇 위치 불러오기
ball_x, ball_y, ball_z = load_xyz('file_pos_data.txt')
robot_x, robot_y, robot_z = load_xyz('file_robot_pos_data.txt')

# Plot
plt.figure(figsize=(10, 8))

# X 위치 비교
plt.subplot(3, 1, 1)
plt.plot(time_data, ball_x, label='Ball X', color='blue')
plt.plot(time_data, robot_x, label='Racket X', color='red', linestyle='--')
plt.ylabel('X Position')
plt.legend()
plt.grid(True)

# Y 위치 비교
plt.subplot(3, 1, 2)
plt.plot(time_data, ball_y, label='Ball Y', color='blue')
plt.plot(time_data, robot_y, label='Racket Y', color='red', linestyle='--')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)

# Z 위치 비교
plt.subplot(3, 1, 3)
plt.plot(time_data, ball_z, label='Ball Z', color='blue')
plt.plot(time_data, robot_z, label='Racket Z', color='red', linestyle='--')
plt.ylabel('Z Position')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show(block=False)

# 라켓 속도 데이터 불러오기
robot_vx, robot_vy, robot_vz = load_xyz('file_robot_vel_data.txt')

# Plot: 라켓 위치 및 속도
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(time_data, robot_x, label='Racket X Pos', color='blue')
plt.plot(time_data, robot_vx, label='Racket X Vel', color='orange')
plt.ylabel('X')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time_data, robot_y, label='Racket Y Pos', color='blue')
plt.plot(time_data, robot_vy, label='Racket Y Vel', color='orange')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time_data, robot_z, label='Racket Z Pos', color='blue')
plt.plot(time_data, robot_vz, label='Racket Z Vel', color='orange')
plt.ylabel('Z')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()