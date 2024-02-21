import numpy as np
from .transform import se3_transform, inverse_transform

# ==================================================================================================================

def load_extrinsic(extrinsics_dir):
  """
  RIGHT_LIDAR_EXTRINSICS = [-0.61153,  0.55676, -0.27023,  0.0027052, -0.041999, -3.1357]
  LEFT_LIDAR_EXTRINSICS  = [-0.60072, -0.34077, -0.26837, -0.0053948, -0.041998, -3.1337]
  RADAR_EXTRINSICS       = [-0.71813,     0.12, -0.54479,          0,      0.05,       0]
  """
  with open(extrinsics_dir + 'velodyne_right.txt') as extrinsics_file:
    RIGHT_LIDAR_EXTRINSICS = [float(x) for x in next(extrinsics_file).split(' ')]
    RIGHT_LIDAR_ROT, RIGHT_LIDAR_POS = se3_transform(RIGHT_LIDAR_EXTRINSICS)
    RIGHT_LIDAR_INV_ROT, RIGHT_LIDAR_INV_POS = inverse_transform(RIGHT_LIDAR_ROT, RIGHT_LIDAR_POS)

  with open(extrinsics_dir + 'velodyne_left.txt') as extrinsics_file:
    LEFT_LIDAR_EXTRINSICS = [float(x) for x in next(extrinsics_file).split(' ')]
    LEFT_LIDAR_ROT, LEFT_LIDAR_POS = se3_transform(LEFT_LIDAR_EXTRINSICS)
    LEFT_LIDAR_INV_ROT, LEFT_LIDAR_INV_POS = inverse_transform(LEFT_LIDAR_ROT, LEFT_LIDAR_POS)

  with open(extrinsics_dir + 'radar.txt') as extrinsics_file:
    RADAR_EXTRINSICS = [float(x) for x in next(extrinsics_file).split(' ')]
    RADAR_ROT, RADAR_POS = se3_transform(RADAR_EXTRINSICS)
    RADAR_INV_ROT, RADAR_INV_POS = inverse_transform(RADAR_ROT, RADAR_POS)

  return RIGHT_LIDAR_ROT, RIGHT_LIDAR_POS, RIGHT_LIDAR_INV_ROT, RIGHT_LIDAR_INV_POS, \
         LEFT_LIDAR_ROT,  LEFT_LIDAR_POS,  LEFT_LIDAR_INV_ROT,  LEFT_LIDAR_INV_POS, \
         RADAR_ROT,       RADAR_POS,       RADAR_INV_ROT,       RADAR_INV_POS


def load_lidar_odometry(data_path):

  lidar_odometry = np.genfromtxt(data_path + 'vo/vo.csv', delimiter=',')[1:]
  lidar_odometry_rot = []
  lidar_odometry_pos = []
  lidar_odometry_timestamp = []
  for sample in lidar_odometry:
    sample_rot, sample_pos = se3_transform(sample[2:])
    lidar_odometry_rot.append(sample_rot)
    lidar_odometry_pos.append(sample_pos)
    lidar_odometry_timestamp.append(int(sample[1]))

  lidar_odometry = {'rot':lidar_odometry_rot, 
                    'pos':lidar_odometry_pos,
                    'timestamp':lidar_odometry_timestamp}
  return lidar_odometry

# ==================================================================================================================

def match_sensor(radar_times, src_times):
 
  dst_times = []
  src_i = 0

  for radar_i in range(len(radar_times)):

    while src_times[src_i] <= radar_times[radar_i]:
      src_i += 1

    if abs(src_times[src_i] - radar_times[radar_i]) > abs(src_times[src_i - 1] - radar_times[radar_i]):
      dst_times.append(int(src_times[src_i - 1]))

    else:
      dst_times.append(int(src_times[src_i]))

  return np.array(dst_times).astype(int)
