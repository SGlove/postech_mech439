import matplotlib.pyplot as plt

# 1. 공 위치 데이터 (file_pos_data.txt): x, y, z
ball_x, ball_y, ball_z = [], [], []
with open('file_pos_data.txt', 'r') as f:
    for line in f:
        x_str, y_str, z_str = line.strip().split(',')
        ball_x.append(float(x_str))
        ball_y.append(float(y_str))
        ball_z.append(float(z_str))

# 2. 시간 데이터 (file_time_data.txt) - 보정
with open('file_time_data.txt', 'r') as f:
    raw_time = [float(line.strip()) for line in f]
    time_data = [t - raw_time[0] for t in raw_time]

# 3. 라켓 각도 데이터 (file_roll_pitch_data.txt): roll, pitch (degree)
roll_data, pitch_data = [], []
with open('file_roll_pitch_data.txt', 'r') as f:
    for line in f:
        roll_str, pitch_str = line.strip().split(',')
        roll_data.append(float(roll_str))
        pitch_data.append(float(pitch_str))

# 4. 그래프 그리기
plt.figure(figsize=(10, 10))

# X
plt.subplot(5, 1, 1)
plt.plot(time_data, ball_x, label='Ball X', color='blue')
plt.ylabel('X (mm)')
plt.grid(True)
plt.legend()

# Y
plt.subplot(5, 1, 2)
plt.plot(time_data, ball_y, label='Ball Y', color='green')
plt.ylabel('Y (mm)')
plt.grid(True)
plt.legend()

# Z
plt.subplot(5, 1, 3)
plt.plot(time_data, ball_z, label='Ball Z', color='red')
plt.ylabel('Z (mm)')
plt.grid(True)
plt.legend()

# Roll
plt.subplot(5, 1, 4)
plt.plot(time_data, roll_data, label='Racket Roll', color='purple')
plt.ylabel('Roll (°)')
plt.grid(True)
plt.legend()

# Pitch
plt.subplot(5, 1, 5)
plt.plot(time_data, pitch_data, label='Racket Pitch', color='orange')
plt.ylabel('Pitch (°)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
