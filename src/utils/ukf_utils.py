####### UKF UTILITIES #######
# IMPORTS
# tools
import pdb
# math
import numpy as np
import numpy.linalg as la
import tf
# libs & utils
from utils.math_utils import *


def state_to_tf(state):
    tf = np.eye(4)
    tf[0:3, 3] = state[0:3]
    tf[0:3, 0:3] = quat_to_rotm(state[6:10])
    return tf


def enforce_pos_def_sym_mat(sigma):
    sigma_out = (sigma + sigma.T) / 2
    eig_vals, eig_vecs = la.eig(sigma_out)
    eig_val_mat = np.diag(np.real(eig_vals))
    eig_vec_mat = np.real(eig_vecs)
    eig_val_mat[eig_val_mat < 0] = 0.000001
    sigma_out = np.matmul(eig_vec_mat, np.matmul(eig_val_mat, eig_vec_mat.T)) ### TEMP PYTHON 2 ###
    # sigma_out = eig_vec_mat @ eig_val_mat @ eig_vec_mat.T
    return sigma_out + 1e-12 * np.eye(sigma_out.shape[0])  # the small addition is for numeric stability


def bb_corners_to_angle(points):
    """
    BB_CORNERS_TO_ANGLE Function that takes takes coordinates of a bounding
    box corners and returns it as center, size and angle.
    points = [x1,y1;x2,y2;x3,y3;x4,y4] (N x 2 matrix)
    """
    sortind_x = np.argsort(points[:, 0], axis=0)  # sort points by x coordinate

    # of the points furthest to the left, which is lower and which is higher? (bottom left / top left)
    if points[sortind_x[0], 1] > points[sortind_x[2], 1]:
        bl_x = points[sortind_x[0], 0]
        bl_y = points[sortind_x[0], 1]
        tl_x = points[sortind_x[1], 0]
        tl_x = points[sortind_x[1], 1]
    else:
        bl_x = points[sortind_x[1], 0]
        bl_y = points[sortind_x[1], 1]
        tl_x = points[sortind_x[0], 0]
        tl_y = points[sortind_x[0], 1]

    # of the points furthest to the right, which is lower and which is higher? (bottom right / top right)
    if points[sortind_x[2], 1] > points[sortind_x[3], 1]:
        br_x = points[sortind_x[2], 0]
        br_y = points[sortind_x[2], 1]
        tr_x = points[sortind_x[3], 0]
        tr_y = points[sortind_x[3], 1]
    else:
        br_x = points[sortind_x[3], 0]
        br_y = points[sortind_x[3], 1]
        tr_x = points[sortind_x[2], 0]
        tr_y = points[sortind_x[2], 1]

    angle = -np.arctan((bl_y - br_y) / (bl_x - br_x))
    x_center = np.mean([br_x, bl_x, tl_x, tr_x])
    y_center = np.mean([br_y, bl_y, tl_y, tr_y])
    width = la.norm([br_x - bl_x, br_y - bl_y])
    height = la.norm([br_x - tr_x, br_y - tr_y])

    return np.array([x_center, y_center, width, height, angle])
