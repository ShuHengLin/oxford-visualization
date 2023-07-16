# Tool to prepare lidar data from Oxford RobotCar dataset.
# Licensed under the Apache License

import tqdm, os
import numpy as np
from lib.transform import se3_transform, inverse_transform, compose_transform, frame_transform

RIGHT_LIDAR_EXTRINSICS = [-0.61153,  0.55676, -0.27023,  0.0027052, -0.041999, -3.1357]
RIGHT_LIDAR_ROT, RIGHT_LIDAR_POS = se3_transform(RIGHT_LIDAR_EXTRINSICS)
RIGHT_LIDAR_INV_ROT, RIGHT_LIDAR_INV_POS = inverse_transform(RIGHT_LIDAR_ROT, RIGHT_LIDAR_POS)

LEFT_LIDAR_EXTRINSICS  = [-0.60072, -0.34077, -0.26837, -0.0053948, -0.041998, -3.1337]
LEFT_LIDAR_ROT, LEFT_LIDAR_POS = se3_transform(LEFT_LIDAR_EXTRINSICS)
LEFT_LIDAR_INV_ROT, LEFT_LIDAR_INV_POS = inverse_transform(LEFT_LIDAR_ROT, LEFT_LIDAR_POS)

RADAR_EXTRINSICS = [-0.71813, 0.12, -0.54479, 0, 0.05, 0]
RADAR_ROT, RADAR_POS = se3_transform(RADAR_EXTRINSICS)
RADAR_INV_ROT, RADAR_INV_POS = inverse_transform(RADAR_ROT, RADAR_POS)

rot_along_x = np.array([[1,  0,  0, 0],
                        [0, -1,  0, 0],
                        [0,  0, -1, 0],
                        [0,  0,  0, 1]])

# ==================================================================================================================

def to_matrix(rot, pos):
  rot = np.matrix(rot.as_matrix())
  pos = np.matrix(pos).T
  rot2 = np.concatenate((rot, [[0, 0, 0]]), axis=0)
  pos2 = np.concatenate((pos, [[1]]), axis=0)
  matrix = np.concatenate((rot2, pos2), axis=1)
  return matrix


def main(data_path):

    lidar_odometry_path = os.path.join(data_path, 'vo/vo.csv')
    lidar_odometry = np.genfromtxt(lidar_odometry_path, delimiter=',')[1:]
    lidar_odometry_rot = []
    lidar_odometry_pos = []
    lidar_odometry_timestamp = []
    for sample in lidar_odometry:
        sample_rot, sample_pos = se3_transform(sample[2:])
        lidar_odometry_rot.append(sample_rot)
        lidar_odometry_pos.append(sample_pos)
        lidar_odometry_timestamp.append(int(sample[1]))
    lidar_odometry_timestamp.append(sample[0])
    lidar_odometry = {'rot':lidar_odometry_rot, 
                      'pos':lidar_odometry_pos,
                      'timestamp':lidar_odometry_timestamp}

    radar_timestamp_path = os.path.join(data_path, 'radar.timestamps')
    radar_timestamp = np.loadtxt(radar_timestamp_path)[:, 0]

    left_lidar_timestamp_path = os.path.join(data_path, 'velodyne_left.timestamps')
    left_lidar_timestamp = np.loadtxt(left_lidar_timestamp_path)[:, 0]

    right_lidar_timestamp_path = os.path.join(data_path, 'velodyne_right.timestamps')
    right_lidar_timestamp = np.loadtxt(right_lidar_timestamp_path)[:, 0]

    lidar_radar_frame_ratio = 5
    right_lidar_timestamp_i = 0
    for radar_timestamp_i in tqdm.tqdm(range(len(radar_timestamp) - 1)):

        while right_lidar_timestamp[right_lidar_timestamp_i] <= radar_timestamp[radar_timestamp_i]:
            right_lidar_timestamp_i += 1

        trans_matrix_list = []
        inv_trans_matrix_list = []
        for right_lidar_timestamp_j in range(right_lidar_timestamp_i-1, right_lidar_timestamp_i+lidar_radar_frame_ratio+1):

            # transform to radar
            frame_rot, frame_pos = frame_transform(right_lidar_timestamp[right_lidar_timestamp_j], radar_timestamp[radar_timestamp_i], lidar_odometry)
            frame_rot, frame_pos = compose_transform(RIGHT_LIDAR_ROT, RIGHT_LIDAR_POS, frame_rot, frame_pos)
            frame_rot, frame_pos = compose_transform(frame_rot, frame_pos, RADAR_INV_ROT, RADAR_INV_POS)

            trans_matrix = to_matrix(frame_rot, frame_pos)
            trans_matrix_list.append(trans_matrix)
        
        matrix_final = np.matrix(np.zeros((4, 4)))
        for matrix_i in range(len(trans_matrix_list)):
          matrix_final += trans_matrix_list[matrix_i]

        matrix_final = (rot_along_x * matrix_final) / 7
        with open('calib/' + str(int(radar_timestamp[radar_timestamp_i])) + '.txt', 'w') as f:
          f.write('lidar_to_radar: ')
          for i in range(4):
            for j in range(4):
              f.write(str(matrix_final[i, j]) + ' ')


if __name__ == '__main__':

    if not os.path.exists('calib'):
      os.mkdir('calib')

    data_path = '/data_1TB_1/Oxford/2019-01-10-11-46-21-radar-oxford-10k'
    main(data_path)
