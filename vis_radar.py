import cv2
import numpy as np
from tqdm import tqdm
from lib.utils_bbox import draw_box_2d
from lib.utils_data_loading import loading_timestamps

scale = 0.2

radar_dir = '/data_1TB_1/Oxford/2019-01-10-11-46-21-radar-oxford-10k/processed/radar/'
radar_timestamps = loading_timestamps(radar_dir)

for i in tqdm(range(10, len(radar_timestamps))):

  # loading radar
  img = cv2.imread(radar_dir + str(radar_timestamps[i]) + '.jpg')


  # loading label
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

    img = draw_box_2d(img, points)

  cv2.namedWindow('img', 0)
  cv2.resizeWindow('img', 1800, 800)
  cv2.imshow('img', img)
  cv2.waitKey(0)

