import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from lib.utils_bbox import compute_box_3d, draw_box_3d, inverse_trans_matrix
from lib.utils_data_loading import loading_timestamps
from lib.camera_model import CameraModel
from lib.image import load_image
from lib.transform import build_se3_transform
from lib.interpolate_poses import interpolate_vo_poses

rot_along_x = np.array([[1,  0,  0, 0],
                        [0, -1,  0, 0],
                        [0,  0, -1, 0],
                        [0,  0,  0, 1]])

width, height = 960, 1280
my_dpi = 80
fig = plt.figure(figsize=(height/my_dpi, width/my_dpi), dpi=my_dpi, frameon=False)
ax = fig.add_subplot()

# ==================================================================================================================

lidar_right_dir = '/data_1TB_1/Oxford/2019-01-10-11-46-21-radar-oxford-10k/velodyne_right/'
lidar_right_timestamps, radar_timestamps = loading_timestamps(lidar_right_dir)

image_dir      = '/data_1TB_1/Oxford/2019-01-10-11-46-21-radar-oxford-10k/stereo/centre/'
models_dir     = '/data_1TB_1/Oxford/robotcar-dataset-sdk/models'
extrinsics_dir = '/data_1TB_1/Oxford/robotcar-dataset-sdk/extrinsics/'

model = CameraModel(models_dir, image_dir)
with open(extrinsics_dir + model.camera + '.txt') as extrinsics_file:
  extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
G_camera_vehicle = build_se3_transform(extrinsics)
G_camera_posesource = G_camera_vehicle

img_timestamps, radar_timestamps = loading_timestamps(image_dir)

for i in tqdm(range(300, len(radar_timestamps))):

    # loading calib
    trans_matrix = np.matrix(np.zeros((4, 4)))
    f = open('/data_1TB_1/Oxford/oxford-visualization/calib/' + str(radar_timestamps[i]) + '.txt', 'r')
    for line in f.readlines():
      line_str = line.strip().split(' ')
      if line_str[0] == 'lidar_to_radar:':
        trans_matrix = np.matrix(line_str[1:], dtype=np.float32).reshape(4, 4)
    inv_trans_matrix = inverse_trans_matrix(trans_matrix)


    # loading transform
    poses_file = '/data_1TB_1/Oxford/2019-01-10-11-46-21-radar-oxford-10k/vo/vo.csv' # vo/vo.csv  gt/radar_odometry.csv
    poses = interpolate_vo_poses(poses_file, [lidar_right_timestamps[i]], img_timestamps[i])

    RIGHT_LIDAR_EXTRINSICS = [-0.61153,  0.55676, -0.27023,  0.0027052, -0.041999, -3.1357]
    G_posesource_laser = build_se3_transform(RIGHT_LIDAR_EXTRINSICS)


    # loading pointcloud
    scan = np.fromfile(lidar_right_dir + str(lidar_right_timestamps[i]) + '.bin', dtype=np.float32).reshape((4, -1))
    scan_new = np.concatenate((scan[0:3], [np.ones(scan.shape[1])]), axis=0)
    scan_new = np.dot(np.dot(poses, G_posesource_laser), scan_new)
    scan[0:3] = scan_new[0:3]


    # loading image
    img = load_image(image_dir + str(img_timestamps[i]) + '.png', model)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # loading label
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
        box_new = np.dot(np.dot(poses, G_posesource_laser), box_new)
        uv, _ = model.project_box(box_new, img.shape)
        if uv.shape[1] == 8:
          img = draw_box_3d(img, np.array(uv.T, dtype=int))
        box_3d = np.array(box_new[0:3])


    vv, depth = model.project(scan_new, img.shape)
#    for x in range(vv.shape[1]):
#      img = cv2.circle(img, (int(vv[0, x]), int(vv[1, x])), radius=2, color=(255, 255, 255), thickness=-1)

    img = model.undistort(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    assert img.shape == (960, 1280, 3)
    ax.clear()
    ax.imshow(img)
    ax.set_xlim(0, 1280)
    ax.set_ylim(960, 0)
    ax.scatter(np.ravel(vv[0, :]), np.ravel(vv[1, :]), s=10, c=depth, edgecolors='none', cmap='jet')
    ax.set_axis_off()
#    fig.savefig('img+.jpg', dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
    plt.pause(0.01)

