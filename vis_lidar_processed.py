import rospy
import numpy as np
from tqdm import tqdm
from lib.utils_pointcloud import compute_box_3d, new_marker_array, box_to_marker
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

# ==================================================================================================================

lidar_dir = '/data_1TB_1/Oxford/2019-01-10-11-46-21-radar-oxford-10k/processed/lidar/'
radar_timestamps = loading_timestamps(lidar_dir)


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

  # loading pointcloud
  scan = np.fromfile(lidar_dir + str(radar_timestamps[i]) + '.bin', dtype=np.float32).reshape((-1, 4)).T
  scan = np.dot(rot_along_x, scan).T
  pointcloud_msg = pcl2.create_cloud(header, fields, scan[:, 0:4])


  # loading label
  marker_array = new_marker_array()
  f = open('/data_1TB_1/Oxford/MVDNet/data/RobotCar/object/label_3d/' + str(radar_timestamps[i]) + '.txt', 'r')
  for i, line in enumerate(f.readlines()):
    line_str = line.strip().split(' ')
    cls = line_str[0]
    dim   = np.array(line_str[8:11], dtype=np.float32)
    loc   = np.array(line_str[11:14], dtype=np.float32)
    rot_y = np.array(line_str[14], dtype=np.float32)

    if cls!= 'DontCare':
      box_3d = compute_box_3d(dim, loc, rot_y)
      marker = box_to_marker(box_3d.T, cls, i)
      marker_array.markers.append(marker)

  marker_pub.publish(marker_array)
  pointcloud_pub.publish(pointcloud_msg)
  rate.sleep()
