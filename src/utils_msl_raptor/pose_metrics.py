#!/usr/bin/env python3
# IMPORTS
# system
import sys, os, time
from copy import copy
from collections import defaultdict
import pdb
# math
import numpy as np
from scipy.spatial.transform import Rotation as R
# Utils
sys.path.append('/root/msl_raptor_ws/src/msl_raptor/src/utils_msl_raptor')
from ssp_utils import *
from math_utils import *
from ros_utils import *

class PoseMetricTracker:
    """
    This class is to help unify how ssp and raptor judge the results. It can be incrementally updated with results at each iteration, 
    and at the end can calculate averages for the run. It calculates several metrics using the methodology from ssp's code.
    """
    def __init__(self, px_thresh=5, prct_thresh=10, trans_thresh=0.05, ang_thresh=5, names=None, bb_3d_dict=None, eps=1e-5):

        self.px_thresh        = px_thresh
        self.prct_thresh      = prct_thresh
        self.trans_thresh     = trans_thresh # meters
        self.ang_thresh       = ang_thresh  # degrees
        self.eps              = eps
        self.names            = names
        self.bb_3d_dict       = bb_3d_dict

        # Init variables
        self.num_measurements    = defaultdict(int)
        self.testing_error_trans = defaultdict(float)
        self.testing_error_angle = defaultdict(float)
        self.testing_error_pixel = defaultdict(float)
        self.errs_2d             = defaultdict(list)
        self.errs_3d             = defaultdict(list)
        self.errs_trans          = defaultdict(list)
        self.errs_angle          = defaultdict(list)
        self.errs_corner2D       = defaultdict(list)

        self.acc                = defaultdict(float)
        self.acc5cm5deg         = defaultdict(float)
        self.acc3d10            = defaultdict(float)
        self.acc5cm5deg         = defaultdict(float)
        self.corner_acc         = defaultdict(float)
        self.mean_err_2d        = defaultdict(float)
        self.mean_err_3d        = defaultdict(float)
        self.mean_corner_err_2d = defaultdict(float)

        # Provide external access to these variables
        self.proj_2d_gt = {}
        self.proj_2d_pr = {}


    def translation_error(self, name, t_cam_ado_gt, t_cam_ado_pr):
        # Compute translation error
        trans_dist = np.sqrt(np.sum(np.square(t_cam_ado_gt - t_cam_ado_pr)))
        self.errs_trans[name].append(trans_dist)
        return trans_dist


    def angle_error(self, name, R_cam_ado_gt, R_cam_ado_pr, tf_w_cam=None):
        # Compute angle error
        if tf_w_cam is not None:
            # do the calculation in world frame
            R_w_ado_gt = tf_w_cam[0:3, 0:3] @ R_cam_ado_gt
            R_w_ado_pr = tf_w_cam[0:3, 0:3] @ R_cam_ado_pr
            angle_dist = calcAngularDistance(R_w_ado_gt, R_w_ado_pr)
            # print(quat_to_ang(rotm_to_quat(R_w_ado_gt).reshape((1,4))))
            # print(quat_to_ang(rotm_to_quat(R_w_ado_pr).reshape((1,4))))
        else:
            # do the calculation in camera frame... i dont think we want this but i will leave it here as an option
            print('WARNING - angle error calculations are done in CAMERA frame (if any angles are "enforced" to 0 this will not be clear')
            angle_dist = calcAngularDistance(R_cam_ado_gt, R_cam_ado_pr)
            # print(quat_to_ang(rotm_to_quat(R_cam_ado_gt).reshape((1,4))))
            # print(quat_to_ang(rotm_to_quat(R_cam_ado_pr).reshape((1,4))))
        self.errs_angle[name].append(angle_dist)
        return angle_dist


    def pixel_error(self, name, vertices, K, Rt_cam_ado_gt=None, Rt_cam_ado_pr=None, R_cam_ado_gt=None, t_cam_ado_gt=None, R_cam_ado_pr=None, t_cam_ado_pr=None):
        # Compute pixel error
        if Rt_cam_ado_gt is None:
            if R_cam_ado_gt is None or t_cam_ado_gt is None:
                raise RuntimeError("Either Rt_cam_ado_gt or both R_cam_ado_gt and t_cam_ado_gt must be provided for pixel error computation")
            Rt_cam_ado_gt = np.concatenate((R_cam_ado_gt, t_cam_ado_gt), axis=1)
        if Rt_cam_ado_pr is None:
            if R_cam_ado_pr is None or t_cam_ado_pr is None:
                raise RuntimeError("Either Rt_cam_ado_pr or both R_cam_ado_pr and t_cam_ado_pr must be provided for pixel error computation")
            Rt_cam_ado_pr = np.concatenate((R_cam_ado_pr, t_cam_ado_pr), axis=1)

        self.proj_2d_gt[name] = compute_projection(vertices, Rt_cam_ado_gt, K)
        self.proj_2d_pr[name] = compute_projection(vertices, Rt_cam_ado_pr, K) 
        norm         = np.linalg.norm(self.proj_2d_gt[name] - self.proj_2d_pr[name], axis=0)
        pixel_dist   = np.mean(norm)
        self.errs_2d[name].append(pixel_dist)
        return pixel_dist


    def corner_2d_error(self, name, vertices=None, corners2D_gt=None, corners2D_pr=None, Rt_cam_ado_gt=None, Rt_cam_ado_pr=None, R_cam_ado_gt=None, t_cam_ado_gt=None, R_cam_ado_pr=None, t_cam_ado_pr=None, K=None):
        """
        Compute corner prediction error
        For gt / pr, provide either corners2D, or vertices & Rt & K, or vertices & R & t & K
        """
        if corners2D_gt is not None:
            pass # we are all good, just use this
        elif vertices is not None or K is None:
            if Rt_cam_ado_gt is not None:
                corners2D_gt = compute_projection(np.hstack((np.reshape([0,0,0,1], (4,1)), vertices)), Rt_cam_ado_gt, K).T
            elif R_cam_ado_gt is not None and t_cam_ado_gt is not None:
                Rt_cam_ado_gt = np.concatenate((R_cam_ado_gt, t_cam_ado_gt), axis=1)
                corners2D_gt = compute_projection(np.hstack((np.reshape([0,0,0,1], (4,1)), vertices)), Rt_cam_ado_gt, K).T
                raise RuntimeError("If corners2D_gt not given, (Rt_cam_ado_gt & K) or (R_cam_ado_gt & t_cam_ado_gt & K) must be provided for 2d corner error computation")
        else:
            raise RuntimeError("Either corners2D_gt OR vertices & tf info must be provide for 2d corner error computation")
        
        if corners2D_pr is not None:
            pass # we are all good, just use this
        elif vertices is not None:
            if Rt_cam_ado_pr is not None:
                corners2D_pr = compute_projection(np.hstack((np.reshape([0,0,0,1], (4,1)), vertices)), Rt_cam_ado_pr, K).T
            elif R_cam_ado_pr is not None and t_cam_ado_pr is not None:
                Rt_cam_ado_pr = np.concatenate((R_cam_ado_pr, t_cam_ado_pr), axis=1)
                corners2D_pr = compute_projection(np.hstack((np.reshape([0,0,0,1], (4,1)), vertices)), Rt_cam_ado_pr, K).T
                raise RuntimeError("If corners2D_pr not given, (Rt_cam_ado_pr & K) or (R_cam_ado_pr & t_cam_ado_pr & K) must be provided for 2d corner error computation")
        else:
            raise RuntimeError("Either corners2D_pr OR vertices & tf info must be provide for 2d corner error computation")
            
        # at this point we have corners2D_gt & corners2D_pr
        
        corner_norm = np.linalg.norm(corners2D_gt - corners2D_pr, axis=1)
        corner_dist = np.mean(corner_norm)
        self.errs_corner2D[name].append(corner_dist)
        return corner_dist


    def corner_3d_error(self, name, vertices, Rt_cam_ado_gt=None, Rt_cam_ado_pr=None, R_cam_ado_gt=None, t_cam_ado_gt=None, R_cam_ado_pr=None, t_cam_ado_pr=None):
        # Compute 3D distances
        if Rt_cam_ado_gt is None:
            if R_cam_ado_gt is None or t_cam_ado_gt is None:
                raise RuntimeError("Either Rt_cam_ado_gt or both R_cam_ado_gt and t_cam_ado_gt must be provided for 3d corner error computation")
            Rt_cam_ado_gt = np.concatenate((R_cam_ado_gt, t_cam_ado_gt), axis=1)
        if Rt_cam_ado_pr is None:
            if R_cam_ado_pr is None or t_cam_ado_pr is None:
                raise RuntimeError("Either Rt_cam_ado_pr or both R_cam_ado_pr and t_cam_ado_pr must be provided for 3d corner error computation")
            Rt_cam_ado_pr = np.concatenate((R_cam_ado_pr, t_cam_ado_pr), axis=1)
        transform_3d_gt   = compute_transformation(vertices, Rt_cam_ado_gt) 
        transform_3d_pr = compute_transformation(vertices, Rt_cam_ado_pr)  
        norm3d            = np.linalg.norm(transform_3d_gt - transform_3d_pr, axis=0)
        vertex_dist       = np.mean(norm3d)
        self.errs_3d[name].append(vertex_dist)
        return vertex_dist


    def update_all_metrics(self, name, vertices, K, tf_w_cam, corners2D_gt=None, corners2D_pr=None, Rt_cam_ado_gt=None, Rt_cam_ado_pr=None, R_cam_ado_gt=None, t_cam_ado_gt=None, R_cam_ado_pr=None, t_cam_ado_pr=None):
        # Sum errors
        if Rt_cam_ado_gt is not None:
            R_cam_ado_gt = Rt_cam_ado_gt[0:3, 0:3]
            t_cam_ado_gt = Rt_cam_ado_gt[0:3, 3]
        elif R_cam_ado_gt is not None and t_cam_ado_gt is not None:
            Rt_cam_ado_gt = np.concatenate((R_cam_ado_gt, t_cam_ado_gt), axis=1)
        else:
            raise RuntimeError("Must provide either Rt_cam_ado_gt or both R_cam_ado_gt and t_cam_ado_gt")

        if Rt_cam_ado_pr is not None:
            R_cam_ado_pr = Rt_cam_ado_pr[0:3, 0:3]
            t_cam_ado_pr = Rt_cam_ado_pr[0:3, 3]
        elif R_cam_ado_pr is not None and t_cam_ado_pr is not None:
            Rt_cam_ado_pr = np.concatenate((R_cam_ado_pr, t_cam_ado_pr), axis=1)
        else:
            raise RuntimeError("Must provide either Rt_cam_ado_pr or both R_cam_ado_pr and t_cam_ado_pr")

        self.testing_error_trans[name] += self.translation_error(name, t_cam_ado_gt, t_cam_ado_pr)
        self.testing_error_angle[name] += self.angle_error(name, R_cam_ado_gt, R_cam_ado_pr, tf_w_cam=tf_w_cam)
        self.corner_2d_error(name, vertices, Rt_cam_ado_gt=Rt_cam_ado_gt, Rt_cam_ado_pr=Rt_cam_ado_pr, K=K)
        self.corner_3d_error(name, vertices, Rt_cam_ado_gt=Rt_cam_ado_gt, Rt_cam_ado_pr=Rt_cam_ado_pr)
        self.testing_error_pixel[name] += self.pixel_error(name, vertices, K, Rt_cam_ado_gt, Rt_cam_ado_pr, R_cam_ado_gt, t_cam_ado_gt, R_cam_ado_pr, t_cam_ado_pr)
        self.num_measurements[name]    += 1


    def calc_final_metrics(self):
        # Compute 2D projection error, 6D pose error, 5cm5degree error

        for name in self.names:
            if self.num_measurements[name] == 0:
                continue
            self.acc[name]          = len(np.where(np.array(self.errs_2d[name]) <= self.px_thresh)[0]) * 100. / (len(self.errs_2d[name]) + self.eps)
            self.acc5cm5deg[name]   = len(np.where((np.array(self.errs_trans[name]) <= self.trans_thresh) & (np.array(self.errs_angle[name]) <= self.ang_thresh))[0]) * 100. / (len(self.errs_trans[name]) + self.eps)
            self.acc3d10[name]      = len(np.where(np.array(self.errs_3d[name]) <= self.bb_3d_dict[name][-1] * self.prct_thresh/100.)[0]) * 100. / (len(self.errs_3d[name]) + self.eps)
            self.acc5cm5deg[name]   = len(np.where((np.array(self.errs_trans[name]) <= self.trans_thresh) & (np.array(self.errs_angle[name]) <= self.ang_thresh))[0]) * 100. / (len(self.errs_trans[name]) + self.eps)
            self.corner_acc[name]   = len(np.where(np.array(self.errs_corner2D[name]) <= self.px_thresh)[0]) * 100. / (len(self.errs_corner2D[name]) + self.eps)
            self.mean_err_2d[name]  = np.mean(self.errs_2d[name])
            self.mean_err_3d[name]  = np.mean(self.errs_3d[name])
            self.mean_corner_err_2d[name] = np.mean(self.errs_corner2D[name])


    def print_final_metrics(self):
        for name in self.names:
            if self.num_measurements[name] == 0:
                continue
            self.print_one_final_metrics(name)


    def print_one_final_metrics(self, name):
        # Print test statistics

        if self.num_measurements[name] == 0:
            print("WARNING: no data was stored in metric calculator for {}".format(name))
            return

        N = float(self.num_measurements[name])
        logging('\nResults of {} -------------------------------'.format(name))
        logging('   Acc using {} px 2D Projection = {:.2f}%'.format(self.px_thresh, self.acc[name]))
        logging('   Acc using {}% threshold - {} vx 3D Transformation = {:.2f}%'.format(self.prct_thresh, self.bb_3d_dict[name][-1] * self.prct_thresh/100, self.acc3d10[name]))
        logging('   Acc using {} cm {} degree metric = {:.2f}%'.format(self.trans_thresh*100, self.ang_thresh, self.acc5cm5deg[name]))
        logging('   Mean 2D pixel error is %f, Mean vertex error is %f, mean corner error is %f' % (self.mean_err_2d[name], self.mean_err_3d[name], self.mean_corner_err_2d[name]))
        logging('   Translation error: %f m, angle error: %f degree, pixel error: %f pix' % (self.testing_error_trans[name]/N, self.testing_error_angle[name]/N, self.testing_error_pixel[name]/N) )
