import matplotlib.pyplot as plt

# 파일 경로
file_pos = 'file_pos_data.txt'
file_vel = 'file_vel_data.txt'
file_time = 'file_time_data.txt'

# 시간 데이터 로딩 및 보정
with open(file_time, 'r') as f:
    time_data = [float(line.strip()) for line in f]
start_time = time_data[0]
time_data = [t - start_time for t in time_data]

# 위치 데이터 로딩 (mm 단위)
x_pos, y_pos, z_pos = [], [], []
with open(file_pos, 'r') as f:
    for line in f:
        x, y, z = map(float, line.strip().split(','))
        x_pos.append(x)
        y_pos.append(y)
        z_pos.append(z)

# 속도 데이터 로딩 (mm/s 단위)
x_vel, y_vel, z_vel = [], [], []
with open(file_vel, 'r') as f:
    for line in f:
        x, y, z = map(float, line.strip().split(','))
        x_vel.append(x)
        y_vel.append(y)
        z_vel.append(z)

# 그래프 1: 위치
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(time_data, x_pos, label='X Position')
plt.ylabel('X (mm)')

plt.subplot(3, 1, 2)
plt.plot(time_data, y_pos, label='Y Position')
plt.ylabel('Y (mm)')

plt.subplot(3, 1, 3)
plt.plot(time_data, z_pos, label='Z Position')
plt.ylabel('Z (mm)')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show(block=False)

# 그래프 2: 속도
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(time_data, x_vel, label='X Velocity', color='r')
plt.ylabel('Vx (mm/s)')

plt.subplot(3, 1, 2)
plt.plot(time_data, y_vel, label='Y Velocity', color='g')
plt.ylabel('Vy (mm/s)')

plt.subplot(3, 1, 3)
plt.plot(time_data, z_vel, label='Z Velocity', color='b')
plt.ylabel('Vz (mm/s)')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show(block=False)

# 그래프 3: Z 위치 vs Z 속도
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(time_data, z_pos, label='Z Position', color='purple')
plt.ylabel('Z (mm)')

plt.subplot(2, 1, 2)
plt.plot(time_data, z_vel, label='Z Velocity', color='blue')
plt.ylabel('Vz (mm/s)')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()
