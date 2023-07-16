import numpy as np
from .transform import se3_transform

# ==================================================================================================================
"""
def match_to_radar(radar_timestamp, src_timestamp):
  dst_timestamps = []
  src_timestamp_i = 0
  for radar_timestamp_i in range(len(radar_timestamp)):
    while src_timestamp[src_timestamp_i] <= radar_timestamp[radar_timestamp_i]:
      src_timestamp_i += 1
    dst_timestamps.append(int(src_timestamp[src_timestamp_i - 1]))
  return dst_timestamps
"""
def match_to_radar(radar_timestamp, src_timestamp):
  dst_timestamps = []
  src_timestamp_i = 0
  for radar_timestamp_i in range(len(radar_timestamp)):
    while src_timestamp[src_timestamp_i] <= radar_timestamp[radar_timestamp_i]:
      src_timestamp_i += 1
    if (src_timestamp[src_timestamp_i] - radar_timestamp[radar_timestamp_i]) > (radar_timestamp[radar_timestamp_i] - src_timestamp[src_timestamp_i - 1]):
      dst_timestamps.append(int(src_timestamp[src_timestamp_i - 1]))
    elif (src_timestamp[src_timestamp_i] - radar_timestamp[radar_timestamp_i]) <= (radar_timestamp[radar_timestamp_i] - src_timestamp[src_timestamp_i - 1]):
      dst_timestamps.append(int(src_timestamp[src_timestamp_i]))
  return dst_timestamps

# ==================================================================================================================

def loading_timestamps(path):

  data_path = path.strip().split('2019-01-10-11-46-21-radar-oxford-10k/')[0]
  data_path += '2019-01-10-11-46-21-radar-oxford-10k/'
  
  sensor = path.strip().split('2019-01-10-11-46-21-radar-oxford-10k/')[1]
  sensor = sensor.strip().split('/')[0]

  radar_timestamp = np.loadtxt(data_path + 'radar.timestamps')[:, 0]

  if sensor == 'velodyne_right':
    right_lidar_timestamp = np.loadtxt(data_path + 'velodyne_right.timestamps')[:, 0]
    return match_to_radar(radar_timestamp, right_lidar_timestamp), radar_timestamp.astype(int)

  elif sensor == 'velodyne_left':
    left_lidar_timestamp = np.loadtxt(data_path + 'velodyne_left.timestamps')[:, 0]
    return match_to_radar(radar_timestamp, left_lidar_timestamp), radar_timestamp.astype(int)

  elif sensor == 'stereo':
    stereo_timestamp = np.loadtxt(data_path + 'stereo.timestamps')[:, 0]
    return match_to_radar(radar_timestamp, stereo_timestamp), radar_timestamp.astype(int)

  else:
    return radar_timestamp.astype(int)

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

