import cv2
import numpy as np
from .utils_pointcloud import rot_X, rot_Y, rot_Z, compute_box_3d, inverse_trans_matrix, points_transform, project_to_image

# ==================================================================================================================

face_idx = [[1, 2, 6, 5],   # front face
            [0, 1, 5, 4],   # left face
            [2, 3, 7, 6],   # back face
            [3, 0, 4, 7]]   # right face

def draw_box_3d(image, corners, c=(0, 255, 0)):

  for ind_f in range(3, -1, -1):
    f = face_idx[ind_f]
    for j in range(4):
      cv2.line(image, (int(corners[f[j],       0]), int(corners[f[j],       1]) ),
                      (int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1]) ), c, 2, lineType=cv2.LINE_AA)
    if ind_f == 1:
      cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1]) ),
                      (int(corners[f[2], 0]), int(corners[f[2], 1]) ), c, 1, lineType=cv2.LINE_AA)
      cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1]) ),
                      (int(corners[f[3], 0]), int(corners[f[3], 1]) ), c, 1, lineType=cv2.LINE_AA)
  return image

# ==================================================================================================================

def draw_box_2d(image, corners, c=(0, 255, 0)):

  cv2.line(image, (corners[0][0], corners[1][0]), (corners[0][1], corners[1][1]), c, 1, lineType=cv2.LINE_AA)
  cv2.line(image, (corners[0][1], corners[1][1]), (corners[0][2], corners[1][2]), c, 1, lineType=cv2.LINE_AA)
  cv2.line(image, (corners[0][2], corners[1][2]), (corners[0][3], corners[1][3]), c, 1, lineType=cv2.LINE_AA)
  cv2.line(image, (corners[0][3], corners[1][3]), (corners[0][0], corners[1][0]), c, 1, lineType=cv2.LINE_AA)

  return image

