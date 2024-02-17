import cv2
import rospy
import numpy as np
from tqdm import tqdm
from lib.utils_bbox import draw_box_3d, draw_box_2d
from lib.utils_pointcloud import compute_box_3d, new_marker_array, box_to_marker, inverse_trans_matrix
from lib.utils_data_loading import loading_timestamps
from lib.camera_model import CameraModel
from lib.image import load_image
from lib.transform import build_se3_transform
from lib.interpolate_poses import interpolate_vo_poses

import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

scale = 0.2

rot_along_x = np.array([[1,  0,  0, 0],
                        [0, -1,  0, 0],
                        [0,  0, -1, 0],
                        [0,  0,  0, 1]])

# ==================================================================================================================

lidar_right_dir = '/data_1TB_1/Oxford/2019-01-10-11-46-21-radar-oxford-10k/velodyne_right/'
lidar_right_timestamps, radar_timestamps = loading_timestamps(lidar_right_dir)

radar_dir = '/data_1TB_1/Oxford/2019-01-10-11-46-21-radar-oxford-10k/processed/radar/'
  
image_dir      = '/data_1TB_1/Oxford/2019-01-10-11-46-21-radar-oxford-10k/stereo/centre/'
models_dir     = '/data_1TB_1/Oxford/robotcar-dataset-sdk/models'
extrinsics_dir = '/data_1TB_1/Oxford/robotcar-dataset-sdk/extrinsics/'
model = CameraModel(models_dir, image_dir)
img_timestamps, radar_timestamps = loading_timestamps(image_dir)


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

for i in tqdm(range(10, len(radar_timestamps))):
    
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
  scan_new = np.dot(rot_along_x, scan_new)
  scan[0:3] = scan_new[0:3]
  pointcloud_msg = pcl2.create_cloud(header, fields, scan.T[:, 0:4])


  # loading lidar label
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
      box_new = np.dot(inv_trans_matrix, box_new)
      box_new = np.dot(rot_along_x, box_new)
      box_3d = np.array(box_new[0:3])
      marker = box_to_marker(box_3d.T, cls, j)
      marker_array.markers.append(marker)


  # loading radar
  radar_img = cv2.imread(radar_dir + str(radar_timestamps[i]) + '.jpg')
    
    
  # loading radar label
  f = open('/data_1TB_1/Oxford/MVDNet/data/RobotCar/object/label_2d/' + str(radar_timestamps[i]) + '.txt', 'r')
  for line in f.readlines():
    line_str = line.strip().split(' ')
    line_float = np.array(line_str[2:], dtype=np.float32)

    width  = line_float[2] / scale
    height = line_float[3] / scale

    # top-left corner
    x_center = (line_float[0] / scale) - (width // 2)
    y_center = (line_float[1] / scale) - (height // 2)

    yaw = line_float[4]
    if line_str[0] == 'Pedestrian':
      yaw = 0

    theta = np.deg2rad(-yaw)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    points = np.array([
      [x_center, y_center],
      [x_center + width, y_center],
      [x_center + width, y_center + height],
      [x_center, y_center + height],
      ]).T

    cx = line_float[0] / scale
    cy = line_float[1] / scale
    T = np.array([[cx], [cy]])
    points = points - T
    points = np.matmul(R, points) + T
    points = points.astype(int)
    points += 160

    radar_img = draw_box_2d(radar_img, points)


  # loading transform
  poses_file = '/data_1TB_1/Oxford/2019-01-10-11-46-21-radar-oxford-10k/gt/radar_odometry.csv'
  poses = interpolate_vo_poses(poses_file, [radar_timestamps[i]], img_timestamps[i])

  RADAR_EXTRINSICS = [-0.71813, 0.12, -0.54479, 0, 0.05, 0]
  G_posesource_radar = build_se3_transform(RADAR_EXTRINSICS)


  # loading image
  img = load_image(image_dir + str(img_timestamps[i]) + '.png', model)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


  # loading camera label
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
      box_new = np.dot(rot_along_x, box_new)
      box_new = np.dot(np.dot(poses, G_posesource_radar), box_new)
      uv, _ = model.project_box(box_new, img.shape)
      if uv.shape[1] == 8:
        img = draw_box_3d(img, np.array(uv.T, dtype=int))
      box_3d = np.array(box_new[0:3])


  # visualize
  marker_pub.publish(marker_array)
  pointcloud_pub.publish(pointcloud_msg)

  radar_img = cv2.resize(radar_img, (960, 960)) 
  img = model.undistort(img)
  img = cv2.copyMakeBorder(img.copy(), 0, 0, 0, 960, cv2.BORDER_CONSTANT, value=[230,230,230])
  img[0:960, 1280:2240] = radar_img

  cv2.namedWindow('img', 0)
  cv2.resizeWindow('img', 1800, 800)
  cv2.imshow('img', img)
  cv2.waitKey(1)
