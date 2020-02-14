#!/usr/bin/env python3
# IMPORTS
# system
import sys, os, time
from copy import copy
import pdb
# math
import numpy as np
from scipy.spatial.transform import Rotation as R
# plots
import matplotlib
# matplotlib.use('Agg')  ## This is needed for the gui to work from a virtual container
import matplotlib.pyplot as plt
# ros
import rosbag
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Pose
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from msl_raptor.msg import AngledBbox, AngledBboxes, TrackedObjects, TrackedObject
import tf
import cv2
from cv_bridge import CvBridge, CvBridgeError
# Utils
sys.path.append('/root/msl_raptor_ws/src/msl_raptor/src/utils_msl_raptor')
from ssp_utils import *
from math_utils import *
from ros_utils import *

class pose_metric_tracker:
    """
    This class is to help unify how ssp and raptor judge the results. It can be incrementally updated with results at each iteration, 
    and at the end can calculate averages for the run. It calculates several metrics using the methodology from ssp's code.
    """
    def __init__(self, px_thresh=5, prct_thresh=10, trans_thresh=0.05, ang_thresh=5, name='mslquad', diam=0.311, eps=1e-5):

        self.px_thresh    = px_thresh
        self.prct_thresh  = prct_thresh
        self.trans_thresh = trans_thresh # meters
        self.ang_thresh   = ang_thresh  # degrees
        self.eps          = eps
        self.name         = name
        self.diam         = diam

        # Init variables
        self.num_measurements    = 0
        self.testing_error_trans = 0.0
        self.testing_error_angle = 0.0
        self.testing_error_pixel = 0.0
        self.errs_2d             = []
        self.errs_3d             = []
        self.errs_trans          = []
        self.errs_angle          = []
        self.errs_corner2D       = []

        self.acc                = None
        self.acc5cm5deg         = None
        self.acc3d10            = None
        self.acc5cm5deg         = None
        self.corner_acc         = None
        self.mean_err_2d        = None
        self.mean_err_3d        = None
        self.mean_corner_err_2d = None

        # Provide external access to these variables
        self.proj_2d_gt = None
        self.proj_2d_pr = None


    def translation_error(self, t_gt, t_pr):
        # Compute translation error
        trans_dist = np.sqrt(np.sum(np.square(t_gt - t_pr)))
        self.errs_trans.append(trans_dist)
        return trans_dist


    def angle_error(self, R_gt, R_pr):
        # Compute angle error
        angle_dist = calcAngularDistance(R_gt, R_pr)
        self.errs_angle.append(angle_dist)
        return angle_dist


    def pixel_error(self, vertices, K, Rt_gt=None, Rt_pr=None, R_gt=None, t_gt=None, R_pr=None, t_pr=None):
        # Compute pixel error
        if Rt_gt is None:
            if R_gt is None or t_gt is None:
                raise RuntimeError("Either Rt_gt or both R_gt and t_gt must be provided for pixel error computation")
            Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
        if Rt_pr is None:
            if R_pr is None or t_pr is None:
                raise RuntimeError("Either Rt_pr or both R_pr and t_pr must be provided for pixel error computation")
            Rt_pr = np.concatenate((R_pr, t_pr), axis=1)

        self.proj_2d_gt   = compute_projection(vertices, Rt_gt, K)
        self.proj_2d_pr = compute_projection(vertices, Rt_pr, K) 
        norm         = np.linalg.norm(self.proj_2d_gt - self.proj_2d_pr, axis=0)
        pixel_dist   = np.mean(norm)
        self.errs_2d.append(pixel_dist)
        return pixel_dist


    def corner_2d_error(self, vertices=None, corners2D_gt=None, corners2D_pr=None, Rt_gt=None, Rt_pr=None, R_gt=None, t_gt=None, R_pr=None, t_pr=None, K=None):
        """
        Compute corner prediction error
        For gt / pr, provide either corners2D, or vertices & Rt & K, or vertices & R & t & K
        """
        if corners2D_gt is not None:
            pass # we are all good, just use this
        elif vertices is not None or K is None:
            if Rt_gt is not None:
                corners2D_gt = compute_projection(np.hstack((np.reshape([0,0,0,1], (4,1)), vertices)), Rt_gt, K).T
            elif R_gt is not None and t_gt is not None:
                Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
                corners2D_gt = compute_projection(np.hstack((np.reshape([0,0,0,1], (4,1)), vertices)), Rt_gt, K).T
                raise RuntimeError("If corners2D_gt not given, (Rt_gt & K) or (R_gt & t_gt & K) must be provided for 2d corner error computation")
        else:
            raise RuntimeError("Either corners2D_gt OR vertices & tf info must be provide for 2d corner error computation")
        
        if corners2D_pr is not None:
            pass # we are all good, just use this
        elif vertices is not None:
            if Rt_pr is not None:
                corners2D_pr = compute_projection(np.hstack((np.reshape([0,0,0,1], (4,1)), vertices)), Rt_pr, K).T
            elif R_pr is not None and t_pr is not None:
                Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
                corners2D_pr = compute_projection(np.hstack((np.reshape([0,0,0,1], (4,1)), vertices)), Rt_pr, K).T
                raise RuntimeError("If corners2D_pr not given, (Rt_pr & K) or (R_pr & t_pr & K) must be provided for 2d corner error computation")
        else:
            raise RuntimeError("Either corners2D_pr OR vertices & tf info must be provide for 2d corner error computation")
            
        # at this point we have corners2D_gt & corners2D_pr
        
        corner_norm = np.linalg.norm(corners2D_gt - corners2D_pr, axis=1)
        corner_dist = np.mean(corner_norm)
        self.errs_corner2D.append(corner_dist)
        return corner_dist


    def corner_3d_error(self, vertices, Rt_gt=None, Rt_pr=None, R_gt=None, t_gt=None, R_pr=None, t_pr=None):
        # Compute 3D distances
        if Rt_gt is None:
            if R_gt is None or t_gt is None:
                raise RuntimeError("Either Rt_gt or both R_gt and t_gt must be provided for 3d corner error computation")
            Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
        if Rt_pr is None:
            if R_pr is None or t_pr is None:
                raise RuntimeError("Either Rt_pr or both R_pr and t_pr must be provided for 3d corner error computation")
            Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
        transform_3d_gt   = compute_transformation(vertices, Rt_gt) 
        transform_3d_pr = compute_transformation(vertices, Rt_pr)  
        norm3d            = np.linalg.norm(transform_3d_gt - transform_3d_pr, axis=0)
        vertex_dist       = np.mean(norm3d)
        self.errs_3d.append(vertex_dist)
        return vertex_dist


    def update_all_metrics(self, vertices, K, corners2D_gt=None, corners2D_pr=None, Rt_gt=None, Rt_pr=None, R_gt=None, t_gt=None, R_pr=None, t_pr=None):
        # Sum errors
        if Rt_gt is not None:
            R_gt = Rt_gt[0:3, 0:3]
            t_gt = Rt_gt[0:3, 3]
        elif R_gt is not None and t_gt is not None:
            Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
        else:
            raise RuntimeError("Must provide either Rt_gt or both R_gt and t_gt")

        if Rt_pr is not None:
            R_pr = Rt_pr[0:3, 0:3]
            t_pr = Rt_pr[0:3, 3]
        elif R_pr is not None and t_pr is not None:
            Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
        else:
            raise RuntimeError("Must provide either Rt_pr or both R_pr and t_pr")

        self.testing_error_trans += self.translation_error(t_gt, t_pr)
        self.testing_error_angle += self.angle_error(R_gt, R_pr)
        self.corner_2d_error(vertices, Rt_gt=Rt_gt, Rt_pr=Rt_pr, K=K)
        self.corner_3d_error(vertices, Rt_gt=Rt_gt, Rt_pr=Rt_pr)
        self.testing_error_pixel += self.pixel_error(vertices, K, Rt_gt, Rt_pr, R_gt, t_gt, R_pr, t_pr)
        self.num_measurements    += 1


    def calc_final_metrics(self):
        # Compute 2D projection error, 6D pose error, 5cm5degree error
       
        self.acc          = len(np.where(np.array(self.errs_2d) <= self.px_thresh)[0]) * 100. / (len(self.errs_2d) + self.eps)
        self.acc5cm5deg   = len(np.where((np.array(self.errs_trans) <= self.trans_thresh) & (np.array(self.errs_angle) <= self.ang_thresh))[0]) * 100. / (len(self.errs_trans) + self.eps)
        self.acc3d10      = len(np.where(np.array(self.errs_3d) <= self.diam * self.prct_thresh/100.)[0]) * 100. / (len(self.errs_3d) + self.eps)
        self.acc5cm5deg   = len(np.where((np.array(self.errs_trans) <= self.trans_thresh) & (np.array(self.errs_angle) <= self.ang_thresh))[0]) * 100. / (len(self.errs_trans) + self.eps)
        self.corner_acc   = len(np.where(np.array(self.errs_corner2D) <= self.px_thresh)[0]) * 100. / (len(self.errs_corner2D) + self.eps)
        self.mean_err_2d  = np.mean(self.errs_2d)
        self.mean_err_3d  = np.mean(self.errs_3d)
        self.mean_corner_err_2d = np.mean(self.errs_corner2D)


    def print_final_metrics(self):
        # Print test statistics
        N = float(self.num_measurements)
        logging('\nResults of {}'.format(self.name))
        logging('   Acc using {} px 2D Projection = {:.2f}%'.format(self.px_thresh, self.acc))
        logging('   Acc using {}% threshold - {} vx 3D Transformation = {:.2f}%'.format(self.prct_thresh, self.diam * self.prct_thresh, self.acc3d10))
        logging('   Acc using {} cm {} degree metric = {:.2f}%'.format(self.trans_thresh*100, self.ang_thresh, self.acc5cm5deg))
        logging('   Mean 2D pixel error is %f, Mean vertex error is %f, mean corner error is %f' % (self.mean_err_2d, self.mean_err_3d, self.mean_corner_err_2d))
        logging('   Translation error: %f m, angle error: %f degree, pixel error: %f pix' % (self.testing_error_trans/N, self.testing_error_angle/N, self.testing_error_pixel/N) )
