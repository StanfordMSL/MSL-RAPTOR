#!/usr/bin/env python3
# IMPORTS
# system
import sys, time
from copy import copy
import pdb
# math
import numpy as np
from scipy.spatial.transform import Rotation as R
# ros
from ssp_utils import *

class raptor_logger:
    def __init__(self, mode="write", names=None, base_path="./", b_ssp=False):
        if names is None:
            raise RuntimeError("Must provide list of names for tracked object")
        self.names = names
        self.base_path = base_path
        self.b_ssp = b_ssp
        self.save_elms = {}
        self.save_elms['est'] = [('Time (s)', 'time', 1),  # list of tuples ("HEADER STRING", "DICT KEY STRING", # OF VALUES (int))
                                 ('Ado State Est', 'state_est', 13), 
                                 ('Ego State Est', 'ego_state_est', 13), 
                                 ('3D Corner Est (X|Y|Z)', 'corners_3d_est', 8*3), 
                                 ('Corner 2D Projections Est (r|c)', 'proj_corners_est', 8*2), 
                                 ('Angled BB (r|c|w|h|ang_deg)', 'abb', 5),
                                 ('Image Segmentation Mode', 'im_seg_mode', 1)]
        self.save_elms['gt'] = [('Time (s)', 'time', 1),  # list of tuples ("HEADER STRING", "DICT KEY STRING", # OF VALUES (int))
                                ('Ado State GT', 'state_gt', 13), 
                                ('Ego State GT', 'ego_state_gt', 13), 
                                ('3D Corner GT (X|Y|Z)', 'corners_3d_gt', 8*3), 
                                ('Corner 2D Projections GT (r|c)', 'proj_corners_gt', 8*2), 
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
        if b_ssp:
            modes = ['est', 'gt', 'prms']
        else:
            modes = ['ssp']
        if mode == "write":
            # create files and write headers
            self.fh = {}
            for n in names:
                for m in modes:
                    # Create logs
                    fn = self.base_path + '_' + n + '_'+ m + '.log'
                    pdb.set_trace()
                    self.create_file_dir(fn)
                    self.fh[m][n] = open(fn,'w+')  # doing this here makes it appendable
                    save_el_shape = (len(self.save_elms[m]), len(self.save_elms[m][0]))
                    data_header = ", ".join(np.reshape([*zip(self.save_elms[m])], save_el_shape)[:,0].tolist())
                    np.savetxt(self.fh[m][n], X=[], header=data_header)  # write header

        elif mode == "read":
            self.fn = {}
            for n in names:
                for m in modes:
                    fn = self.base_path + '_' + n + '_'+ m + '.log'
                    self.fn[m][n] = fn
        else:
            raise RuntimeError("Unrecognized logging mode")


    def write_data_to_log(self, data, name, mode='est'):
        """ mode can be est, gt, ssp, or param"""
        if (not self.b_ssp and not mode in ['est', 'gt', 'prms']) or (self.b_ssp and not mode == 'ssp'):
            raise RuntimeError("Mode {} not recognized".format(mode))
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
        np.savetxt(self.fh[mode][name], X=out, fmt='%.6f')  # write to file


    def read_logs(self, name):
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

            data = np.loadtxt(self.fn[log_type][name])
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
            for n in self.names:
                self.fh[fh_key][n].close()


    def create_file_dir(self, fn_with_dir):
        path = "/".join(fn_with_dir.split("/")[:-1])
        if not os.path.exists( path ):
            os.makedirs( path )
