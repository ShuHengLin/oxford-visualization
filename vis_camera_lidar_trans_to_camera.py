import cv2
import numpy as np
from tqdm import tqdm

from lib.utils_bbox import compute_box_3d, inverse_trans_matrix, points_transform, project_to_image, draw_box_3d
from lib.utils_data_loading import load_extrinsic, load_lidar_odometry, match_sensor
from lib.transform import compose_transform, frame_transform, to_matrix

from sdk.python.camera_model import CameraModel
from sdk.python.image import load_image

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

label_dir  = base_dir + 'oxford-visualization/MVDNet/data/RobotCar/object/label_3d/'
image_dir  = data_path + 'stereo/centre/'
models_dir = base_dir + 'oxford-visualization/sdk/models/'

model = CameraModel(models_dir, image_dir)

image_times = np.loadtxt(data_path + 'stereo.timestamps'        )[:, 0].astype(int)
lidar_times = np.loadtxt(data_path + 'velodyne_right.timestamps')[:, 0].astype(int)
radar_times = np.loadtxt(data_path + 'radar.timestamps'         )[:, 0].astype(int)[:8862]

align_times     = match_sensor(image_times, lidar_times)  # align lidar to camera
align_times_cam = match_sensor(radar_times, image_times)
align_times_lid = match_sensor(radar_times, align_times)

# ==================================================================================================================

R = np.array([[ 0,  1,  0, 0],
              [ 0,  0, -1, 0],
              [-1,  0,  0, 0],
              [ 0,  0,  0, 1]]) # np.dot(rot_X(90), rot_Z(-90))

# ==================================================================================================================

for i in tqdm(range(820, len(radar_times))):

  # lidar to camera transform
  frame_rot, frame_pos = frame_transform(align_times_lid[i], align_times_cam[i], lidar_odometry)
  frame_rot, frame_pos = compose_transform(RIGHT_LIDAR_ROT, RIGHT_LIDAR_POS, frame_rot, frame_pos)
  trans_matrix1 = to_matrix(frame_rot, frame_pos)


  # loading image
  img = load_image(image_dir + str(align_times_cam[i]) + '.png', model)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


  # lidar to radar transform
  frame_rot, frame_pos = frame_transform(align_times_lid[i], radar_times[i], lidar_odometry)
  frame_rot, frame_pos = compose_transform(RIGHT_LIDAR_ROT, RIGHT_LIDAR_POS, frame_rot, frame_pos)
  frame_rot, frame_pos = compose_transform(frame_rot, frame_pos, RADAR_INV_ROT, RADAR_INV_POS)
  trans_matrix = to_matrix(frame_rot, frame_pos)
  trans_matrix = np.dot(R, trans_matrix) # transform to radar


  # loading label
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

    uv, _ = project_to_image(box_3d, model.G_camera_image, model.focal_length, model.principal_point, img.shape)
    if uv.shape[1] == 8:
      img = draw_box_3d(img, np.array(uv.T, dtype=int))

#  cv2.imwrite('img.jpg', img)
  cv2.imshow('img', img)
  cv2.waitKey()
