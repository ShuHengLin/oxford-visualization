import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from lib.utils_bbox import points_transform, project_to_image
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
lidar_dir  = data_path + 'velodyne_right/'
models_dir = base_dir + 'oxford-visualization/sdk/models/'

model = CameraModel(models_dir, image_dir)

image_times = np.loadtxt(data_path + 'stereo.timestamps'        )[:, 0].astype(int)
lidar_times = np.loadtxt(data_path + 'velodyne_right.timestamps')[:, 0].astype(int)
align_times = match_sensor(image_times, lidar_times)

# ==================================================================================================================

width, height = 960, 1280
my_dpi = 80

fig = plt.figure(figsize=(height/my_dpi, width/my_dpi), dpi=my_dpi, frameon=False)
ax = fig.add_subplot()

# ==================================================================================================================

for i in tqdm(range(5000, len(image_times))):

  # lidar to camera transform
  frame_rot, frame_pos = frame_transform(align_times[i], image_times[i], lidar_odometry)
  frame_rot, frame_pos = compose_transform(RIGHT_LIDAR_ROT, RIGHT_LIDAR_POS, frame_rot, frame_pos)
  trans_matrix = to_matrix(frame_rot, frame_pos)


  # loading pointcloud
  scan = np.fromfile(lidar_dir + str(align_times[i]) + '.bin', dtype=np.float32).reshape((4, -1))
  scan[0:3] = points_transform(scan[0:3], trans_matrix)
  scan = scan[:, scan[0, :] > 0]  # crop by front


  # loading image
  img = load_image(image_dir + str(image_times[i]) + '.png', model)


  # project lidar points on image
  uv, depth = project_to_image(scan, model.G_camera_image, model.focal_length, model.principal_point, img.shape)
  assert img.shape == (width, height, 3)
  ax.clear()
  ax.imshow(img)
  ax.set_xlim(0, height)
  ax.set_ylim(width, 0)
  ax.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=10, c=depth, edgecolors='none', cmap='jet')
  ax.set_axis_off()
#  fig.savefig('img.jpg', dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
#  plt.show()
  plt.pause(0.01)
