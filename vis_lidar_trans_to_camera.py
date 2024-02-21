import rospy
import numpy as np
from tqdm import tqdm
from lib.utils_pointcloud import compute_box_3d, inverse_trans_matrix, points_transform, new_marker_array, box_to_marker
from lib.utils_data_loading import load_extrinsic, load_lidar_odometry, match_sensor
from lib.transform import compose_transform, frame_transform, to_matrix

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
lidar_odometry = load_lidar_odometry(data_path)

RIGHT_LIDAR_ROT, RIGHT_LIDAR_POS, RIGHT_LIDAR_INV_ROT, RIGHT_LIDAR_INV_POS, \
LEFT_LIDAR_ROT,  LEFT_LIDAR_POS,  LEFT_LIDAR_INV_ROT,  LEFT_LIDAR_INV_POS, \
RADAR_ROT,       RADAR_POS,       RADAR_INV_ROT,       RADAR_INV_POS \
 = load_extrinsic(base_dir + 'oxford-visualization/sdk/extrinsics/')

label_dir = base_dir + 'oxford-visualization/MVDNet/data/RobotCar/object/label_3d/'
image_dir = data_path + 'stereo/centre/'
lidar_dir = data_path + 'velodyne_right/'

image_times = np.loadtxt(data_path + 'stereo.timestamps'        )[:, 0].astype(int)
lidar_times = np.loadtxt(data_path + 'velodyne_right.timestamps')[:, 0].astype(int)
radar_times = np.loadtxt(data_path + 'radar.timestamps'         )[:, 0].astype(int)[:8862]

align_times     = match_sensor(image_times, lidar_times)  # align lidar to camera
align_times_cam = match_sensor(radar_times, image_times)
align_times_lid = match_sensor(radar_times, align_times)

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

  # lidar to camera transform
  frame_rot, frame_pos = frame_transform(align_times_lid[i], align_times_cam[i], lidar_odometry)
  frame_rot, frame_pos = compose_transform(RIGHT_LIDAR_ROT, RIGHT_LIDAR_POS, frame_rot, frame_pos)
  trans_matrix1 = to_matrix(frame_rot, frame_pos)


  # loading pointcloud
  scan = np.fromfile(lidar_dir + str(align_times_lid[i]) + '.bin', dtype=np.float32).reshape((4, -1))
  scan[0:3] = points_transform(scan[0:3], trans_matrix1)
#  scan = scan[:, scan[0, :] > 0]  # crop by front
  pointcloud_msg = pcl2.create_cloud(header, fields, scan.T[:, 0:4])


  # lidar to radar transform
  frame_rot, frame_pos = frame_transform(align_times_lid[i], radar_times[i], lidar_odometry)
  frame_rot, frame_pos = compose_transform(RIGHT_LIDAR_ROT, RIGHT_LIDAR_POS, frame_rot, frame_pos)
  frame_rot, frame_pos = compose_transform(frame_rot, frame_pos, RADAR_INV_ROT, RADAR_INV_POS)
  trans_matrix = to_matrix(frame_rot, frame_pos)
  trans_matrix = np.dot(R, trans_matrix) # transform to radar


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
    box_3d = points_transform(box_3d, inverse_trans_matrix(trans_matrix))
    box_3d = points_transform(box_3d, trans_matrix1)
    marker = box_to_marker(box_3d.T, cls, j)
    marker_array.markers.append(marker)

  marker_pub.publish(marker_array)
  pointcloud_pub.publish(pointcloud_msg)
  rate.sleep()
