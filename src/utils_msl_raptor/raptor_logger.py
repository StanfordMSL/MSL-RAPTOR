#!/usr/bin/env python3
# IMPORTS
# system
import sys, time
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
import tf
import cv2
from cv_bridge import CvBridge, CvBridgeError
from ssp_utils import *

class raptor_logger:
    """
    This class is a interface for working with the custom raptor logfiles. Data can be written / read through this class.
    There are defined log files currently supported - est (for raptor's estimates), gt (raptors ground truth), prms (for parameters), 
    and ssp (for the baseline). These logs will be read by the result_analyzer, and allow the different methods to be compared.
    """
    def __init__(self, source='MSLRAPTOR', mode="write", est_fn=None, gt_fn=None, param_fn=None):
        self.source = source  # MSLRAPTOR, SSP, GT

        self.save_elms = {}
        self.save_elms['est'] = [('Time (s)', 'time', 1),  # list of tuples ("HEADER STRING", "DICT KEY STRING", # OF VALUES (int))
                                 ('Ado State Est', 'state_est', 13), 
                                 ('Ego State Est', 'ego_state_est', 13), 
                                 ('3D Corner Est (X|Y|Z)', 'corners_3d_est', 8*3), 
                                 ('Corner 2D Projections (r|c)', 'proj_corners_est', 8*2), 
                                 ('Angled BB (r|c|w|h|ang_deg)', 'abb', 5),
                                 ('Image Segmentation Mode', 'im_seg_mode', 1)]
        self.save_elms['gt'] = [('Time (s)', 'time', 1),  # list of tuples ("HEADER STRING", "DICT KEY STRING", # OF VALUES (int))
                                ('Ado State GT', 'state_gt', 13), 
                                ('Ego State GT', 'ego_state_gt', 13), 
                                ('3D Corner GT (X|Y|Z)', 'corners_3d_gt', 8*3), 
                                ('Corner 2D Projections (r|c)', 'proj_corners_gt', 8*2), 
                                ('Angled BB (r|c|w|h|ang_deg)', 'abb', 5),
                                ('Image Segmentation Mode', 'im_seg_mode', 1)]
        self.save_elms['prms'] = [('Camera Intrinsics (K)', 'K', 4),
                                  ('Object BB Size (len|wid|hei|diam)', '3d_bb_dims', 4),
                                  ('tf_cam_ego', 'tf_cam_ego', 16)]
        self.save_elms['ssp'] = [('Time (s)', 'time', 1),  # list of tuples ("HEADER STRING", "DICT KEY STRING", # OF VALUES (int))
                                 ('Ado State GT', 'state_gt', 13), 
                                 ('Ado State Est', 'state_est', 13), 
                                 ('Ego State Est', 'ego_state_est', 13), 
                                 ('Ego State GT', 'ego_state_gt', 13), 
                                 ('3D Corner Est (X|Y|Z)', 'corners_3d_gt', 8*3), 
                                 ('3D Corner GT (X|Y|Z)', 'corners_3d_gt', 8*3), 
                                 ('Corner 2D Projections Est (r|c)', 'proj_corners_est', 8*2), 
                                 ('Corner 2D Projections GT (r|c)', 'proj_corners_gt', 8*2)]
        self.log_data = None
        self.fh = None
        self.fn = None
        if mode == "write":
            # create files and write headers
            self.fh = {}
            if est_fn is not None:
                self.create_file_dir(est_fn)
                self.fh['est'] = open(est_fn,'w+')  # doing this here makes it appendable
                save_el_shape = (len(self.save_elms['est']), len(self.save_elms['est'][0]))
                data_header = ", ".join(np.reshape([*zip(self.save_elms['est'])], save_el_shape)[:,0].tolist())
                np.savetxt(self.fh['est'], X=[], header=data_header)  # write header
            if gt_fn is not None:
                self.create_file_dir(gt_fn)
                self.fh['gt'] = open(gt_fn,'w+')  # doing this here makes it appendable
                save_el_shape = (len(self.save_elms['gt']), len(self.save_elms['gt'][0]))
                data_header = ", ".join(np.reshape([*zip(self.save_elms['gt'])], save_el_shape)[:,0].tolist())
                np.savetxt(self.fh['gt'], X=[], header=data_header)  # write header
            if param_fn is not None:
                self.create_file_dir(param_fn)
                self.fh['prms'] = open(param_fn,'w+')  # doing this here makes it appendable
                save_el_shape = (len(self.save_elms['prms']), len(self.save_elms['prms'][0]))
                data_header = ", ".join(np.reshape([*zip(self.save_elms['prms'])], save_el_shape)[:,0].tolist())
                np.savetxt(self.fh['prms'], X=[], header=data_header)  # write header
        elif mode == "read":
            self.fn = {}
            if est_fn is not None:
                self.fn['est'] = est_fn
            if gt_fn is not None:
                self.fn['gt'] = gt_fn
            if param_fn is not None:
                self.fn['prms'] = param_fn
        else:
            raise RuntimeError("Unrecognized logging mode")


    def write_data_to_log(self, data, mode='est'):
        """ mode can be est, gt, ssp, or param"""
        if not mode in ['est', 'gt', 'ssp', 'prms']:
            raise RuntimeError("Mode {} not recognized (must be 'est', 'gt', 'ssp', or 'prms')".format(mode))
        t = None  # time (seconds)
        state = None  # 13 el pos, lin vel, quat, ang vel [ling/ ang vel might be NaN]
        corners_3d_cam_frame = None
        proj_corners = None
        abb_2d = None
        save_el_shape = (len(self.save_elms[mode]), len(self.save_elms[mode][0]))
        num_to_write = np.sum(np.reshape([*zip(self.save_elms[mode])], save_el_shape)[:,2].astype(int)) 
        out = np.ones((1, num_to_write)) * 1e10
        ind = 0
        for i, (header_str, dict_str, count) in enumerate(self.save_elms[mode]):
            if dict_str in data:
                try:
                    out[0, ind:(ind + count)] = data[dict_str]
                except:
                    print("issue with {}".format(dict_str))
                    pdb.set_trace()
            ind += count
        out[out>1e5] = np.nan
        np.savetxt(self.fh[mode], X=out, fmt='%.6f')  # write to file


    def read_logs(self):
        """
        Return a dict with keys being log type (est /gt /prms). Each of these is a dict with the various types of values in the log
        """
        log_data = {}
        for log_type in self.fn:
            if not log_type in self.save_elms:
                print("Warning: we are are missing the log file for {}".format(log_type))
                continue  
            log_data[log_type] = {}
            ind = 0

            data = np.loadtxt(self.fn[log_type])
            for i, (header_str, dict_str, count) in enumerate(self.save_elms[log_type]):
                if len(data.shape) > 1:
                    log_data[log_type][dict_str] = data[:, ind:(ind + count)]
                else:
                    log_data[log_type][dict_str] = data[ind:(ind + count)]
                ind += count
            
                if log_type == 'prms' and dict_str == 'K': # Turn camera intrinsics back into a matrix
                    K = np.eye(3)
                    K[0, 0] = log_data[log_type][dict_str][0]
                    K[1, 1] = log_data[log_type][dict_str][1]
                    K[0, 2] = log_data[log_type][dict_str][2]
                    K[1, 2] = log_data[log_type][dict_str][3]
                    log_data[log_type][dict_str] = K
        self.log_data = log_data
        return log_data
        

    def close_files(self):
        for fh_key in self.fh:
            self.fh[fh_key].close()


    def create_file_dir(self, fn_with_dir):
        path = "/".join(fn_with_dir.split("/")[:-1])
        if not os.path.exists( path ):
            os.makedirs( path )
