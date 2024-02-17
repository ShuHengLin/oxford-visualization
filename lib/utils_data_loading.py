import numpy as np
from .transform import se3_transform

# ==================================================================================================================

def match_to_radar(radar_time, src_time):
 
  dst_time = []
  src_i = 0

  for radar_i in range(len(radar_time)):

    while src_time[src_i] <= radar_time[radar_i]:
      src_i += 1

    if abs(src_time[src_i] - radar_time[radar_i]) > abs(src_time[src_i - 1] - radar_time[radar_i]):
      dst_time.append(int(src_time[src_i - 1]))

    else:
      dst_time.append(int(src_time[src_i]))

  return dst_time

# ==================================================================================================================

def loading_timestamps(path):

  data_path = path.strip().split('2019-01-10-11-46-21-radar-oxford-10k/')[0]
  data_path += '2019-01-10-11-46-21-radar-oxford-10k/'
  
  sensor = path.strip().split('2019-01-10-11-46-21-radar-oxford-10k/')[1]
  sensor = sensor.strip().split('/')[0]

  radar_time = np.loadtxt(data_path + 'radar.timestamps')[:, 0].astype(int)

  if sensor == 'velodyne_right':
    right_lidar_time = np.loadtxt(data_path + 'velodyne_right.timestamps')[:, 0].astype(int)
    return match_to_radar(radar_time, right_lidar_time), radar_time

  elif sensor == 'velodyne_left':
    left_lidar_time = np.loadtxt(data_path + 'velodyne_left.timestamps')[:, 0].astype(int)
    return match_to_radar(radar_time, left_lidar_time), radar_time

  elif sensor == 'stereo':
    stereo_time = np.loadtxt(data_path + 'stereo.timestamps')[:, 0].astype(int)
    return match_to_radar(radar_time, stereo_time), radar_time

  else:
    return radar_time

# ==================================================================================================================

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

