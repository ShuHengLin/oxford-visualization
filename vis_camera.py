import cv2
import numpy as np
from tqdm import tqdm
from lib.utils_bbox import compute_box_3d, draw_box_3d
from lib.utils_data_loading import loading_timestamps
from lib.camera_model import CameraModel
from lib.image import load_image
from lib.transform import build_se3_transform
from lib.interpolate_poses import interpolate_vo_poses

rot_along_x = np.array([[1,  0,  0, 0],
                        [0, -1,  0, 0],
                        [0,  0, -1, 0],
                        [0,  0,  0, 1]])

# ==================================================================================================================

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

    # loading transform
    poses_file = '/data_1TB_1/Oxford/2019-01-10-11-46-21-radar-oxford-10k/gt/radar_odometry.csv' # vo/vo.csv  gt/radar_odometry.csv
    poses = interpolate_vo_poses(poses_file, [radar_timestamps[i]], img_timestamps[i])

    RADAR_EXTRINSICS = [-0.71813, 0.12, -0.54479, 0, 0.05, 0]
    G_posesource_radar = build_se3_transform(RADAR_EXTRINSICS)


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
        box_new = np.dot(rot_along_x, box_new)
        box_new = np.dot(np.dot(poses, G_posesource_radar), box_new)
        uv, _ = model.project_box(box_new, img.shape)
        if uv.shape[1] == 8:
          img = draw_box_3d(img, np.array(uv.T, dtype=int))
        box_3d = np.array(box_new[0:3])


    img = model.undistort(img)
#    cv2.imwrite('img.jpg', img)
    cv2.imshow('img', img)
    cv2.waitKey(0)

