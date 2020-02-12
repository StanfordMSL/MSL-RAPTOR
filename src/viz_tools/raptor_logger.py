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
from msl_raptor.msg import AngledBbox, AngledBboxes, TrackedObjects, TrackedObject
import tf
import cv2
from cv_bridge import CvBridge, CvBridgeError
# from ssp_utils import *


class raptor_logger:
    def __init__(self, source='MSLRAPTOR', est_fn=None, gt_fn=None, param_fn=None):
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
                                  ('Object BB Size (len|wid|hei)', '3d_bb_dims', 3)]
        # save_el_shape = (len(self.save_elements), len(self.save_elements[0]))
        # self.num_to_write = np.sum(np.reshape([*zip(self.save_elements)], save_el_shape)[:,2].astype(int)) 
        
        # create files and write headers
        self.fh = {}
        if est_fn is not None:
            self.fh['est'] = open(est_fn,'w+')  # doing this here makes it appendable
            save_el_shape = (len(self.save_elms['est']), len(self.save_elms['est'][0]))
            # self.num_to_write = np.sum(np.reshape([*zip(self.save_elements)], save_el_shape)[:,2].astype(int)) 
            data_header = ", ".join(np.reshape([*zip(self.save_elms['est'])], save_el_shape)[:,0].tolist())
            np.savetxt(self.fh['est'], X=[], header=data_header)  # write header
        if gt_fn is not None:
            self.fh['gt'] = open(gt_fn,'w+')  # doing this here makes it appendable
            save_el_shape = (len(self.save_elms['gt']), len(self.save_elms['gt'][0]))
            data_header = ", ".join(np.reshape([*zip(self.save_elms['gt'])], save_el_shape)[:,0].tolist())
            np.savetxt(self.fh['gt'], X=[], header=data_header)  # write header
        if param_fn is not None:
            self.fh['prms'] = open(param_fn,'w+')  # doing this here makes it appendable
            save_el_shape = (len(self.save_elms['prms']), len(self.save_elms['prms'][0]))
            data_header = ", ".join(np.reshape([*zip(self.save_elms['prms'])], save_el_shape)[:,0].tolist())
            np.savetxt(self.fh['prms'], X=[], header=data_header)  # write header


    def write_data_to_log(self, data, mode='est'):
        """ mode can be est, gt, or param"""
        if not mode in ['est', 'gt', 'prms']:
            raise RuntimeError("Mode {} not recognized (must be 'est', 'gt', or 'prms')".format(mode))
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

    def close_files(self):
        for fh_key in self.fh:
            self.fh[fh_key].close()