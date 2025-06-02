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
plt.figure(figsize=(10, 12))

# 공 X vs 로봇 X
plt.subplot(6, 1, 1)
plt.plot(time_data, ball_x, label='Ball X', color='blue')
plt.ylabel('Ball X')
plt.grid(True)

plt.subplot(6, 1, 2)
plt.plot(time_data, robot_x, label='Racket X', color='red')
plt.ylabel('Racket X')
plt.grid(True)

# 공 Y vs 로봇 Y
plt.subplot(6, 1, 3)
plt.plot(time_data, ball_y, label='Ball Y', color='blue')
plt.ylabel('Ball Y')
plt.grid(True)

plt.subplot(6, 1, 4)
plt.plot(time_data, robot_y, label='Racket Y', color='red')
plt.ylabel('Racket Y')
plt.grid(True)

# 공 Z vs 로봇 Z
plt.subplot(6, 1, 5)
plt.plot(time_data, ball_z, label='Ball Z', color='blue')
plt.ylabel('Ball Z')
plt.grid(True)

plt.subplot(6, 1, 6)
plt.plot(time_data, robot_z, label='Racket Z', color='red')
plt.ylabel('Racket Z')
plt.xlabel('Time (s)')
plt.grid(True)

plt.tight_layout()
plt.show()