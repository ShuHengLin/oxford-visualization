import rospy
import numpy as np
from tqdm import tqdm
from lib.utils_pointcloud import compute_box_3d, points_transform, new_marker_array, box_to_marker

import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray

##############################
# Options
##############################

base_dir = '/data_1TB_2/Oxford/'

##############################
# End of Options
##############################

data_path = base_dir + '2019-01-10-11-46-21-radar-oxford-10k/'

label_dir = base_dir + 'oxford-visualization/MVDNet/data/RobotCar/object/label_3d/'
lidar_dir = data_path + 'processed/lidar/'

radar_times = np.loadtxt(data_path + 'radar.timestamps')[:, 0].astype(int)[:8862]

# ==================================================================================================================

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

R = np.array([[ 0,  1,  0, 0],
              [ 0,  0, -1, 0],
              [-1,  0,  0, 0],
              [ 0,  0,  0, 1]]) # np.dot(rot_X(90), rot_Z(-90))

# ==================================================================================================================

for i in tqdm(range(600, len(radar_times))):

  if rospy.is_shutdown():
    break

  # loading pointcloud
  scan = np.fromfile(lidar_dir + str(radar_times[i]) + '.bin', dtype=np.float32).reshape((-1, 4)).T
  scan[0:3] = points_transform(scan[0:3], R) # transform to radar
  pointcloud_msg = pcl2.create_cloud(header, fields, scan.T[:, 0:4])


  # loading label
  marker_array = new_marker_array()
  f = open(label_dir + str(radar_times[i]) + '.txt', 'r')
  for j, line in enumerate(f.readlines()):
    line_str = line.strip().split(' ')
    cls = line_str[0]
    dim   = np.array(line_str[8:11],  dtype=np.float32)
    loc   = np.array(line_str[11:14], dtype=np.float32)
    rot_y = np.array(line_str[14],    dtype=np.float32)
    box_3d = compute_box_3d(dim, loc, rot_y)
    marker = box_to_marker(box_3d.T, cls, j)
    marker_array.markers.append(marker)

  marker_pub.publish(marker_array)
  pointcloud_pub.publish(pointcloud_msg)
  rate.sleep()
