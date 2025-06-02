import matplotlib.pyplot as plt

# 파일 경로
velocity_file = 'file_velocity_data.txt'
time_file = 'file_time_data.txt'
z_pos_file = 'file_z_pos_data.txt'

# 시간 데이터 불러오기 및 보정
with open(time_file, 'r') as f:
    time_data = [float(line.strip()) for line in f.readlines()]
base_time = time_data[0]
time_data = [t - base_time for t in time_data]

# 속도 데이터 불러오기
vx, vy, vz = [], [], []
with open(velocity_file, 'r') as f:
    for line in f:
        x, y, z = map(float, line.strip().split(','))
        vx.append(x)
        vy.append(y)
        vz.append(z)

# Z 좌표 데이터 불러오기
with open(z_pos_file, 'r') as f:
    z_pos = [float(line.strip()) for line in f.readlines()]

# 1. 기존 속도 그래프 (3개 축)
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time_data, vx, color='r', label='Vx')
plt.ylabel('Velocity X')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time_data, vy, color='g', label='Vy')
plt.ylabel('Velocity Y')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time_data, vz, color='b', label='Vz')
plt.xlabel('Time (s)')
plt.ylabel('Velocity Z')
plt.grid(True)

plt.tight_layout()
plt.show()

# 2. Z좌표와 Z속도 그래프
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(time_data, z_pos, color='purple', label='Z Position')
plt.ylabel('Position Z')
plt.title('Z-axis Position over Time')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_data, vz, color='blue', label='Velocity Z')
plt.xlabel('Time (s)')
plt.ylabel('Velocity Z')
plt.title('Z-axis Velocity over Time')
plt.grid(True)

plt.tight_layout()
plt.show()