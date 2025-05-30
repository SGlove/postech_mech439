'''
pos #1
x: 917.6005445081958y: 442.12963182060014z: 445.6496841631681
pos #2
x: 790.0315752939126y: 751.6119646095926z: 414.3427606629201
pos #3
x: 502.81341418569093y: 700.69745401055z: 399.8831392324978
pos #4
x: 397.4528855932461y: 494.4052639078356z: 426.0253670565719
pos #5
x: 912.2814033676029y: 445.9636008098544z: 581.791938885872
pos #6
x: 786.872151482024y: 753.260335456152z: 510.3133581243761
pos #7
x: 501.6494466993015y: 702.4246827361914z: 481.1063989638217
pos #8
x: 393.47249766584605y: 496.24059788698685z: 548.2268612980452

width = 300
height = 100

pos #1 -> (150, 150, 50)
pos #2 -> (-150, 150, 50)
pos #3 -> (-150, -150, 50)
pos #4 -> (150, -150, 50)
pos #5 -> (150, 150, -50)
pos #6 -> (-150, 150, -50)
pos #7 -> (-150, -150, -50)
pos #8 -> (150, -150, -50)
'''

import numpy as np

def calculate_affine_transform(robot_points, camera_points):
    """
    로봇 좌표계 -> 카메라 좌표계로 변환하는 3D 아핀 변환 행렬 계산
    :param robot_points: 로봇 좌표계 점들 [(x1, y1, z1), (x2, y2, z2), ...] (8개 점)
    :param camera_points: 카메라 좌표계 대응점들 [(x1, y1, z1), (x2, y2, z2), ...] (8개 점)
    :return: 4x4 아핀 변환 행렬
    """
    # 동차 좌표계로 변환 (로봇 좌표계)
    A = []
    for x, y, z in robot_points:
        A.append([x, y, z, 1])
    A = np.array(A)  # 8x4 행렬
    
    # 카메라 좌표계 (타겟)
    B = np.array(camera_points)  # 8x3 행렬
    
    # 최소 제곱법으로 변환 행렬 계산 (A * M = B)
    M, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
    
    # 4x4 변환 행렬 구성
    T = np.eye(4)
    T[:3, :] = M.T  # 상위 3x4 부분 적용
    
    return T

def transform_point(T, point):
    """
    변환 행렬을 사용해 점 변환
    :param T: 4x4 변환 행렬
    :param point: 변환할 점 (x, y, z)
    :return: 카메라 좌표계의 점 (x, y, z)
    """
    homogeneous_point = np.array([point[0], point[1], point[2], 1])
    transformed = T @ homogeneous_point
    return transformed[:3]  # 동차 좌표 -> 3D 좌표

# ======================== 사용 예시 ========================
if __name__ == "__main__":
    # 로봇 좌표계의 직육면체 8개 점 (±w/2, ±l/2, ±h/2)
    w, l, h = 300, 300, 100  # 예시 크기 (가로, 세로, 높이)
    robot_points = [
        ( w/2,  l/2,  h/2),  # 1
        ( -w/2,  l/2, h/2),  # 2
        ( -w/2, -l/2,  h/2),  # 3
        ( w/2, -l/2, h/2),  # 4
        (w/2,  l/2,  -h/2),  # 5
        (-w/2,  l/2, -h/2),  # 6
        (-w/2, -l/2, -h/2),  # 7
        (w/2, -l/2, -h/2)   # 8
    ]
    
    # 카메라 좌표계에서 측정된 대응점들 (예시 데이터 - 실제 측정값으로 교체 필요)
    # 주의: 실제 구현시 실제 카메라에서 측정한 값을 사용해야 함!
    camera_points = [
        (918, 442, 446),   # 1에 대응
        (790, 752, 414),  # 2에 대응
        (503,701,400),  # 3에 대응
        (398, 446, 582), # 4에 대응
        (912, 446, 582),  # 5에 대응
        (787, 753, 510), # 6에 대응
        (502,702, 481), # 7에 대응
        (393, 496, 548) # 8에 대응
    ]
    
    # 변환 행렬 계산
    T = calculate_affine_transform(robot_points, camera_points)
    print("계산된 4x4 변환 행렬:")
    print(np.round(T, 4))
    
    # 테스트: 로봇 좌표계 (w/6, l/3, 0) -> 카메라 좌표계 변환
    test_point = (w/6, l/3, 0)
    transformed_point = transform_point(T, test_point)
    print(f"\n로봇 좌표 {test_point} -> 카메라 좌표: {np.round(transformed_point, 4)}")