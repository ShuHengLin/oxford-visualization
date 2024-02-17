import rospy
import numpy as np
from tqdm import tqdm
from lib.utils_pointcloud import compute_box_3d, new_marker_array, box_to_marker, inverse_trans_matrix
from lib.utils_data_loading import loading_timestamps

import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

rot_along_x = np.array([[1,  0,  0, 0],
                        [0, -1,  0, 0],
                        [0,  0, -1, 0],
                        [0,  0,  0, 1]])

TRANS_TO_RADAR_MAP = False

# ==================================================================================================================

lidar_right_dir = '/data_1TB_1/Oxford/2019-01-10-11-46-21-radar-oxford-10k/velodyne_right/'
lidar_right_timestamps, radar_timestamps = loading_timestamps(lidar_right_dir)


header = std_msgs.msg.Header()
header.frame_id = 'map'

fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
          PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
          PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
          PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)]

pointcloud_pub = rospy.Publisher('/pointcloud',   PointCloud2, queue_size=10)
marker_pub     = rospy.Publisher('/detect_box3d', MarkerArray, queue_size=10)
rospy.init_node('talker', anonymous=True)
rate = rospy.Rate(1000)

# ==================================================================================================================

for i in tqdm(range(300, len(radar_timestamps))):
    
  if rospy.is_shutdown():
    break

  # loading calib
  trans_matrix = np.matrix(np.zeros((4, 4)))
  f = open('/data_1TB_1/Oxford/oxford-visualization/calib/' + str(radar_timestamps[i]) + '.txt', 'r')
  for line in f.readlines():
    line_str = line.strip().split(' ')
    if line_str[0] == 'lidar_to_radar:':
      trans_matrix = np.matrix(line_str[1:], dtype=np.float32).reshape(4, 4)
  inv_trans_matrix = inverse_trans_matrix(trans_matrix)


  # loading pointcloud
  scan = np.fromfile(lidar_right_dir + str(lidar_right_timestamps[i]) + '.bin', dtype=np.float32).reshape((4, -1))
  scan_new = np.concatenate((scan[0:3], [np.ones(scan.shape[1])]), axis=0)
  if TRANS_TO_RADAR_MAP:     
    scan_new = np.dot(trans_matrix, scan_new)
  else:
    scan_new = np.dot(rot_along_x, scan_new)
  scan[0:3] = scan_new[0:3]
  pointcloud_msg = pcl2.create_cloud(header, fields, scan.T[:, 0:4])


  # loading label
  marker_array = new_marker_array()
  f = open('/data_1TB_1/Oxford/MVDNet/data/RobotCar/object/label_3d/' + str(radar_timestamps[i]) + '.txt', 'r')
  for j, line in enumerate(f.readlines()):
    line_str = line.strip().split(' ')
    cls = line_str[0]
    dim   = np.array(line_str[8:11], dtype=np.float32)
    loc   = np.array(line_str[11:14], dtype=np.float32)
    rot_y = np.array(line_str[14], dtype=np.float32)

    if cls!= 'DontCare':
      box_3d = compute_box_3d(dim, loc, rot_y)
      box_new = np.concatenate((box_3d, [np.ones(box_3d.shape[1])]), axis=0)
      if not TRANS_TO_RADAR_MAP:
        box_new = np.dot(inv_trans_matrix, box_new)
        box_new = np.dot(rot_along_x, box_new)
      box_3d = np.array(box_new[0:3])
      marker = box_to_marker(box_3d.T, cls, j)
      marker_array.markers.append(marker)

  marker_pub.publish(marker_array)
  pointcloud_pub.publish(pointcloud_msg)
  rate.sleep()
