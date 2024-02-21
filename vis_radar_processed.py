import cv2
import numpy as np
from tqdm import tqdm
from lib.utils_bbox import compute_box_3d, draw_box_2d

##############################
# Options
##############################

base_dir = '/data_1TB_2/Oxford/'

scale = 0.2

##############################
# End of Options
##############################

data_path = base_dir + '2019-01-10-11-46-21-radar-oxford-10k/'

label_dir = base_dir + 'oxford-visualization/MVDNet/data/RobotCar/object/label_3d/'
radar_dir = data_path + 'processed/radar/'

radar_times = np.loadtxt(data_path + 'radar.timestamps')[:, 0].astype(int)[:8862]

for i in tqdm(range(600, len(radar_times))):

  # loading radar
  img = cv2.imread(radar_dir + str(radar_times[i]) + '.jpg')


  # loading 3D label
  f = open(label_dir + str(radar_times[i]) + '.txt', 'r')
  for j, line in enumerate(f.readlines()):
    line_str = line.strip().split(' ')
    cls = line_str[0]
    dim   = np.array(line_str[8:11],  dtype=np.float32)
    loc   = np.array(line_str[11:14], dtype=np.float32)
    rot_y = np.array(line_str[14],    dtype=np.float32)
    box_3d = compute_box_3d(dim, loc, rot_y)

    box_3d = np.delete(box_3d, 1, 0).T
    box_3d += [32, 32]
    box_3d /= 64
    box_3d *= 320
    img = draw_box_2d(img, np.array(box_3d).astype(int).T, (0, 0, 255))


  # loading 2D label
  f = open(base_dir + 'oxford-visualization/MVDNet/data/RobotCar/object/label_2d/' + str(radar_times[i]) + '.txt', 'r')
  for line in f.readlines():
    line_str = line.strip().split(' ')
    line_float = np.array(line_str[2:], dtype=np.float32)

    width  = line_float[2] / scale
    height = line_float[3] / scale

    # top-left corner
    x_top_left = (line_float[0] / scale) - (width  // 2)
    y_top_left = (line_float[1] / scale) - (height // 2)

    points = np.array([[x_top_left,         y_top_left         ],
                       [x_top_left + width, y_top_left         ],
                       [x_top_left + width, y_top_left + height],
                       [x_top_left,         y_top_left + height]]).T

    yaw = line_float[4]
    theta = np.deg2rad(-yaw)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    cx = line_float[0] / scale
    cy = line_float[1] / scale
    T = np.array([[cx], [cy]])
    points = points - T
    points = np.matmul(R, points) + T
    points = points.astype(int)
    points += 160
    img = draw_box_2d(img, points)

  cv2.namedWindow('img', 0)
  cv2.resizeWindow('img', 800, 800)
  cv2.imshow('img', img)
  cv2.waitKey()
