#!/usr/bin/env python3
# IMPORTS
# system
import sys, time
from copy import copy
import pdb
# math
import numpy as np
# opencv
import cv2
from cv_bridge import CvBridge, CvBridgeError
# Utils
# sys.path.append('/root/msl_raptor_ws/src/msl_raptor/src/utils_msl_raptor')
# from ros_utils import find_closest_by_time
# from ssp_utils import *



def draw_2d_proj_of_3D_bounding_box(open_cv_image, corners2D_pr, corners2D_gt=None, color_pr=(0,0,255), linewidth=1, inds_to_connect=[[0, 3], [3, 2], [2, 1], [1, 0], [7, 4], [4, 5], [5, 6], [6, 7], [3, 4], [2, 5], [0, 7], [1, 6]], b_verts_only=False):
    """
    corners2D_gt/corners2D_pr is a 8x2 numpy array
    corner order (from ado frame):
        # 1 front, left,  up
        # 2 front, right, up
        # 3 back,  right, up
        # 4 back,  left,  up
        # 5 front, left,  down
        # 6 front, right, down
        # 7 back,  right, down
        # 8 back,  left,  down
    """

    dot_radius = 2
    color_list = [(255, 0, 0),     # 0 blue: center
                    (0, 255, 0),     # 1 green: front lower right
                    (0, 0, 255),     # 2 red: front upper right
                    (255, 255, 0),   # 3 cyan: front lower left
                    (255, 0, 255),   # 4 magenta: front upper left
                    (0, 255, 255),   # 5 yellow: back lower right
                    (0, 0, 0),       # 6 black: back upper right
                    (255, 255, 255), # 7 white: back lower left
                    (125, 125, 125)] # 8 grey: back upper left
    # 

    color_gt = (255,0,0)  # blue
    # color_pr = (0,0,255)  # red
    corners2D_pr = np.round(corners2D_pr).astype(int)
    # pdb.set_trace()
    if b_verts_only:
        # if corners2D_gt is not None:
        #     for i, pnt in enumerate(corners2D_gt):
        #         open_cv_image = cv2.circle(open_cv_image, (pnt[0], pnt[1]), dot_radius, color_list[i], -1)  # -1 means filled in, else edge thickness
        if corners2D_pr is not None:
            for i, pnt in enumerate(corners2D_pr):
                open_cv_image = cv2.circle(open_cv_image, (pnt[0], pnt[1]), dot_radius, color_list[i % len(color_list)], -1)  # -1 means filled in, else edge thickness
    else:
        for inds in inds_to_connect:
            if corners2D_gt is not None:
                open_cv_image = cv2.line(open_cv_image, (corners2D_gt[inds[0],0], corners2D_gt[inds[0],1]), (corners2D_gt[inds[1],0], corners2D_gt[inds[1], 1]), color_gt, linewidth)
            open_cv_image = cv2.line(open_cv_image, (corners2D_pr[inds[0],0], corners2D_pr[inds[0],1]), (corners2D_pr[inds[1],0], corners2D_pr[inds[1], 1]), color_pr, linewidth)

    return open_cv_image