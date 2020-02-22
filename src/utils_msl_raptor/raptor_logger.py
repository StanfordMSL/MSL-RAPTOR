#!/usr/bin/env python3
# IMPORTS
# system
import sys, time
from copy import copy
from collections import defaultdict
import pdb
# math
import numpy as np
from scipy.spatial.transform import Rotation as R
# ros
from ssp_utils import *

class RaptorLogger:
    """
    This helper class writes to /reads from log files. 

    * save_elms is a class var that defines what variables will be in the log files. There are different ones for estimation, ground truth, ssp, and params
    * to write, the user will pass in an object name and a dict with keys corresponding to the second element of each tuple in save_elms
    * to read the user gives the object name, and a dict is passed back
    * params are treated slightly differently, with their own read/write functions
    """
    def __init__(self, mode="write", names=None, base_path="./", b_ssp=False):

        self.names = names
        self.base_path = base_path
        self.b_ssp = b_ssp
        self.save_elms = {}
        
        self.log_data = defaultdict(dict)
        self.fh = None
        self.fn = None
        self.prm_fn = self.base_path + '_prms.log'

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
        self.save_elms['err'] = [('Time (s)', 'time', 1),  # list of tuples ("HEADER STRING", "DICT KEY STRING", # OF VALUES (int))
                                 ('x err', 'x_err', 1),
                                 ('y err', 'y_err', 1),
                                 ('z err', 'z_err', 1),
                                 ('ang err (deg)', 'ang_err', 1),
                                 ('3d projection pix norm err', 'pix_err', 1),
                                 ('3d corner norm err (ADD)', 'add_err', 1),
                                 ('Distance camera to ado', 'measurement_dist', 1)]
        self.save_elms['ssp'] = [('Time (s)', 'time', 1),  # list of tuples ("HEADER STRING", "DICT KEY STRING", # OF VALUES (int))
                                 ('Ado State GT', 'state_gt', 13), 
                                 ('Ado State Est', 'state_est', 13), 
                                 ('Ego State Est', 'ego_state_est', 13), 
                                 ('Ego State GT', 'ego_state_gt', 13), 
                                 ('3D Corner Est (X|Y|Z)', 'corners_3d_gt', 8*3), 
                                 ('3D Corner GT (X|Y|Z)', 'corners_3d_gt', 8*3), 
                                 ('Corner 2D Projections Est (r|c)', 'proj_corners_est', 8*2), 
                                 ('Corner 2D Projections GT (r|c)', 'proj_corners_gt', 8*2)]
        self.save_elms['ssperr'] = self.save_elms['err']

        if not b_ssp:
            self.modes = ['est', 'gt', 'err']
        else:
            self.modes = ['ssp', 'ssperr']


        if mode=="read":
            self.init_read()
        elif mode=="write":
            if names is None:
                raise RuntimeError("Must provide list of names for tracked object")
            self.init_write()
        else:
            raise RuntimeError("Unrecognized logging mode")


    def init_write(self):
        all_name_str = ''
        for n in self.names:
            all_name_str += (n + ' ')
        all_name_str = all_name_str[:-1]
        self.save_elms['prms'] = [('Camera Intrinsics (K)', 'K', 4),
                                    ('tf_cam_ego', 'tf_cam_ego', 16),
                                    ('Object BB Size (len|wid|hei|diam) [{}]'.format(all_name_str), '3d_bb_dims', 4*len(self.names))]

        # create files and write headers
        self.fh = defaultdict(dict)
        for m in self.modes:
            for n in self.names:
                # Create logs
                fn = self.base_path + '_' + n + '_'+ m + '.log'
                self.create_file_dir(fn)
                self.fh[m][n] = open(fn,'w+')  # doing this here makes it appendable
                save_el_shape = (len(self.save_elms[m]), len(self.save_elms[m][0]))
                data_header = ", ".join(np.reshape([*zip(self.save_elms[m])], save_el_shape)[:,0].tolist())
                np.savetxt(self.fh[m][n], X=[], header=data_header)  # write header
        

    def init_read(self):
        self.save_elms['prms'] = [('Camera Intrinsics (K)', 'K', 4),
                                    ('tf_cam_ego', 'tf_cam_ego', 16),
                                    ('Object BB Size (len|wid|hei|diam) []', '3d_bb_dims', -1)]
        self.read_params()
        self.fn = defaultdict(dict)
        for m in self.modes:
            if self.names is None:
                return
            for n in self.names:
                self.fn[m][n] = self.base_path + '_' + n + '_'+ m + '.log'


    def write_params(self, param_data, mode='prms'):
        # write header
        self.create_file_dir(self.prm_fn)
        prm_fh = open(self.prm_fn,'w+')  # doing this here makes it appendable
        save_el_shape = (len(self.save_elms[mode]), len(self.save_elms[mode][0]))
        data_header = ", ".join(np.reshape([*zip(self.save_elms[mode])], save_el_shape)[:,0].tolist())
        np.savetxt(prm_fh, X=[], header=data_header)  # write header
        
        # write body
        save_el_shape = (len(self.save_elms[mode]), len(self.save_elms[mode][0]))
        num_to_write = np.sum(np.reshape([*zip(self.save_elms[mode])], save_el_shape)[:,2].astype(int)) 
        out = np.ones((1, num_to_write)) * 1e10
        ind = 0
        for i, (header_str, dict_str, count) in enumerate(self.save_elms[mode]):
            if dict_str in param_data:
                try:
                    out[0, ind:(ind + count)] = param_data[dict_str]
                except:
                    print("issue with {}".format(dict_str))
                    pdb.set_trace()
            ind += count
        out[out>1e5] = np.nan
        np.savetxt(prm_fh, X=out, fmt='%.6f')  # write to file
        prm_fh.close()

    
    def read_params(self, log_type='prms'):
        # get header
        if not os.path.isfile(self.prm_fn):
            print("WARNING: NOT READING PARAMS")
            return
        f = open(self.prm_fn)
        header_str = f.readline()
        self.log_data[log_type]['ado_names'] = header_str.split('[')[1].split(']')[0].split(' ')
        self.names = self.log_data[log_type]['ado_names']

        # Read rest of file
        data = np.loadtxt(self.prm_fn)
        f.close()
        ind = 0
        for i, (header_str, dict_str, count) in enumerate(self.save_elms[log_type]):
            if len(data.shape) > 1:
                self.log_data[log_type][dict_str] = data[:, ind:(ind + count)]
            else:
                self.log_data[log_type][dict_str] = data[ind:(ind + count)]
            ind += count
            if dict_str == 'K': # Turn camera intrinsics back into a matrix
                K = np.eye(3)
                K[0, 0] = self.log_data[log_type][dict_str][0]
                K[1, 1] = self.log_data[log_type][dict_str][1]
                K[0, 2] = self.log_data[log_type][dict_str][2]
                K[1, 2] = self.log_data[log_type][dict_str][3]
                self.log_data[log_type][dict_str] = K
            elif dict_str == 'tf_cam_ego':
                self.log_data[log_type][dict_str] = np.reshape(self.log_data[log_type][dict_str], (4, 4))
            elif dict_str == '3d_bb_dims':
                all_sizes = np.asarray(data[ind : ind + 4*len(self.log_data[log_type]['ado_names'])])
                bb_3d_dict_all = {}
                for k, name in enumerate(self.log_data[log_type]['ado_names']):
                    bb_3d_dict_all[name] = all_sizes[4*k : 4*k+4]  # len|wid|hei|diam
                self.log_data[log_type][dict_str] = bb_3d_dict_all
        return self.log_data[log_type]


    def write_data_to_log(self, data, name, mode):
        """ mode can be est, gt, ssp"""
        if not self.b_ssp and not mode in self.modes:
            raise RuntimeError("Mode {} not recognized. Available modes are {}".format(mode, self.modes))
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
        for log_type in self.fn:
            if not log_type in self.save_elms:
                print("Warning: we are are missing the log file for {}".format(log_type))
                continue
            ind = 0
            data = np.loadtxt(self.fn[log_type][name])
            for i, (header_str, dict_str, count) in enumerate(self.save_elms[log_type]):
                if len(data.shape) > 1:
                    self.log_data[log_type][dict_str] = data[:, ind:(ind + count)]
                else:
                    self.log_data[log_type][dict_str] = data[ind:(ind + count)]
                ind += count
                
        return self.log_data


    def read_err_logs(self, log_path):
        """
        Return a dict with keys being error type
        """
        err_log_dict = {}
        ind = 0
        f = open(log_path)
        header_str = f.readline()
        data = np.loadtxt(f)
        for i, (header_str, dict_str, count) in enumerate(self.save_elms["err"]):
            if len(data.shape) > 1:
                err_log_dict[dict_str] = data[:, ind:(ind + count)]
            else:
                err_log_dict[dict_str] = data[ind:(ind + count)]
            ind += count
                
        return err_log_dict
        

    def close_files(self):
        for fh_key in self.fh:
            if fh_key == 'prms':
                self.fh[fh_key].close()
                continue
            for n in self.names:
                self.fh[fh_key][n].close()


    def create_file_dir(self, fn_with_dir):
        path = "/".join(fn_with_dir.split("/")[:-1])
        if not os.path.exists( path ):
            os.makedirs( path )
