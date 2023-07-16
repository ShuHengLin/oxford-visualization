import numpy as np
import rospy

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# ==================================================================================================================

# clearing all markers / view in RVIZ remotely
#https://answers.ros.org/question/53595/clearing-all-markers-view-in-rviz-remotely/

def new_marker_array():
  marker_array_msg = MarkerArray()
  marker = Marker()
  marker.id = 0
  marker.action = Marker.DELETEALL
  marker_array_msg.markers.append(marker)
  return marker_array_msg
  
# ==================================================================================================================

def rot_X(degree):
  c, s = np.cos(degree * np.pi / 180), np.sin(degree * np.pi / 180)
  R = np.array([[1,  0,  0],
                [0,  c, -s],
                [0,  s,  c]], dtype=np.float32)
  return R


def rot_Y(degree):
  c, s = np.cos(degree * np.pi / 180), np.sin(degree * np.pi / 180)
  R = np.array([[ c, 0, s],
                [ 0, 1, 0],
                [-s, 0, c]], dtype=np.float32)
  return R


def rot_Z(degree):
  c, s = np.cos(degree * np.pi / 180), np.sin(degree * np.pi / 180)
  R = np.array([[c, -s, 0],
                [s,  c, 0],
                [0,  0, 1]], dtype=np.float32)
  return R

# ==================================================================================================================
              
#https://en.wikipedia.org/wiki/Rotation_matrix

def compute_box_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(-rotation_y), np.sin(-rotation_y)
  R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

  LOCATION = np.matrix([location[2], location[1], -location[0]]).T
  LOCATION = np.dot(rot_X(270), LOCATION)

  l, h, w = dim[0], dim[1], dim[2]
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
  corners_3d = np.vstack([x_corners, y_corners, z_corners])
  corners_3d = np.dot(rot_X(270), corners_3d)

  corners_3d_cam2 = np.dot(R, corners_3d)
  corners_3d_cam2 += LOCATION

  return corners_3d_cam2

# ==================================================================================================================

def color_select(cls, marker):

    if cls == 'Car':
      marker.color.r = 0      # Green
      marker.color.g = 1
      marker.color.b = 0

    elif cls == 'Pedestrian':
      marker.color.r = 1      # Red
      marker.color.g = 0
      marker.color.b = 0

    elif cls == 'Cyclist':
      marker.color.r = 1      # Yellow
      marker.color.g = 1
      marker.color.b = 0

    elif cls == 'Truck':
      marker.color.r = 0      # Cyan
      marker.color.g = 1
      marker.color.b = 1

    elif cls == 'Van':
      marker.color.r = 1      # Purple
      marker.color.g = 0
      marker.color.b = 1

    else:
      marker.color.r = 1      # White
      marker.color.g = 1
      marker.color.b = 1

    return marker

# ==================================================================================================================

lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
         [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

def box_to_marker(ob, cls, i):

  detect_points_set = []
  for x in range(8):
    detect_points_set.append(Point(ob[x][0], ob[x][1], ob[x][2]))

  marker = Marker()
  marker.header.frame_id = 'map'
  marker.header.stamp = rospy.Time.now()
  marker.id = i
  marker.action = Marker.ADD
  marker.type = Marker.LINE_LIST
  marker.lifetime = rospy.Duration(0)

  marker = color_select(cls, marker)
  marker.color.a = 1
  marker.scale.x = 0.2
  marker.points = []

  for line in lines:
    marker.points.append(detect_points_set[line[0]])
    marker.points.append(detect_points_set[line[1]])

  return marker

# ==================================================================================================================

# https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html

def inverse_trans_matrix(trans_matrix):
  inv_trans_rot = np.linalg.inv(trans_matrix[0:3, 0:3])
  inv_trans_rot = np.concatenate((inv_trans_rot, [[0, 0, 0]]), axis=0)

  inv_trans_pos = -np.linalg.inv(trans_matrix[0:3, 0:3]) * trans_matrix[0:3, 3]
  inv_trans_pos = np.concatenate((inv_trans_pos, [[1]]), axis=0)

  return np.concatenate((inv_trans_rot, inv_trans_pos), axis=1)

