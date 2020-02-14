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
from raptor_logger import *
from pose_metrics import *

class result_analyser:

    def __init__(self, log_identifier, ego_quad_ns="/quad7", ado_quad_ns="/quad4"):
        us_split = log_identifier.split("_")
        if log_identifier[-4:] == '.bag' or "_".join(us_split[0:3]) == 'msl_raptor_output':
            # This means id is the source rosbag name for the log files
            if log_identifier[-4:] == '.bag':
                log_base_name = "log_" + us_split[-1][:-4]
            else:
                log_base_name = "log_" + us_split[-1]
        elif log_identifier[0:3] == 'log':
            # we assume this is the log's name (w/o EST/GT/PARAMS etc)
            log_base_name = "_".join(us_split[:-1])
        else:
            raise RuntimeError("We do not recognize log file! {} not understood".format(log_identifier))

        log_in_dir = '/mounted_folder/raptor_logs'
        est_log_name = log_in_dir + "/" + log_base_name + "_EST.log"
        gt_log_name = log_in_dir + "/" + log_base_name + "_GT.log"
        param_log_name = log_in_dir + "/" + log_base_name + "_PARAM.log"
        log_in_dir = '/mounted_folder/ssp_logs'
        ssp_log_name = log_in_dir + "/" + log_base_name + "_SSP.log"
        if not os.path.isfile(ssp_log_name):
            ssp_log_name = None
        self.logger = raptor_logger(source="MSLRAPTOR", mode="read", est_fn=est_log_name, gt_fn=gt_log_name, param_fn=param_log_name, ssp_fn=ssp_log_name)
        self.b_degrees = True  # use degrees or radians

        self.fig = None
        self.axes = None
        self.name = 'mslquad'

        # data to extract
        self.est_data = None
        self.gt_data = None
        self.t0 = -1
        self.tf = -1
        self.t_est = []
        self.t_gt = []
        self.x_est = []
        self.x_gt = []
        self.y_est = []
        self.y_gt = []
        self.z_est = []
        self.z_gt = []
        self.roll_est = []
        self.roll_gt = []
        self.pitch_est = []
        self.pitch_gt = []
        self.yaw_est = []
        self.yaw_gt = []

        self.ego_gt_time_pose = []
        self.ego_gt_pose = []
        self.ego_est_pose = []
        self.ego_est_time_pose = []
        self.ado_gt_pose = []
        self.ado_est_pose = []

        self.DETECT = 1
        self.TRACK = 2
        self.FAKED_BB = 3
        self.IGNORE = 4
        self.detect_time = []
        self.detect_mode = []
        
        self.detect_times = []
        self.detect_end = None

        self.ssp_data = None
        self.t_ssp     = []
        self.x_ssp     = []
        self.y_ssp     = []
        self.z_ssp     = []
        self.roll_ssp  = []
        self.pitch_ssp = []
        self.yaw_ssp   = []
        self.ssp_ado_est_pose = None

        # params to extract
        self.prm_data = None
        self.K = None
        self.dist_coefs = None
        self.new_camera_matrix = None
        self.prm_data = None
        self.est_data = None
        self.gt_data = None
        self.box_length = None
        self.box_width = None
        self.box_height = None
        self.raptor_metrics = pose_metric_tracker()
        #################################

        self.extract_logs()
        self.do_plot()
        self.quant_eval()


    def extract_logs(self):
        # read the data
        log_data = self.logger.read_logs()
        if 'prms' in log_data:
            self.prm_data = log_data['prms']

            # extract params
            self.new_camera_matrix = self.prm_data['K']
            self.tf_cam_ego = np.reshape(self.prm_data['tf_cam_ego'], (4, 4))
            self.box_length, self.box_width, self.box_height, self.diam = list(self.prm_data['3d_bb_dims'])
            self.vertices = np.array(  [[ self.box_length/2, self.box_width/2, self.box_height/2, 1.],
                                        [ self.box_length/2, self.box_width/2,-self.box_height/2, 1.],
                                        [ self.box_length/2,-self.box_width/2,-self.box_height/2, 1.],
                                        [ self.box_length/2,-self.box_width/2, self.box_height/2, 1.],
                                        [-self.box_length/2,-self.box_width/2, self.box_height/2, 1.],
                                        [-self.box_length/2,-self.box_width/2,-self.box_height/2, 1.],
                                        [-self.box_length/2, self.box_width/2,-self.box_height/2, 1.],
                                        [-self.box_length/2, self.box_width/2, self.box_height/2, 1.]]).T

        # extract est
        if 'est' in log_data:
            self.est_data = log_data['est']
            self.t_est = self.est_data['time']
            self.tf = self.t_est[-1]
            ado_state_est_mat = self.est_data['state_est']
            self.x_est = ado_state_est_mat[:, 0]
            self.y_est = ado_state_est_mat[:, 1]
            self.z_est = ado_state_est_mat[:, 2]
            rpy_mat = quat_to_ang(ado_state_est_mat[:, 6:10])
            self.roll_est = rpy_mat[:, 0]
            self.pitch_est = rpy_mat[:, 1]
            self.yaw_est = rpy_mat[:, 2]
            
            self.ego_est_time_pose = self.est_data['time']  # use same times
            
            n = len(ado_state_est_mat)
            self.ado_est_pose = np.concatenate((quat_to_rotm(ado_state_est_mat[:, 6:10]), 
                                                np.expand_dims(ado_state_est_mat[:, 0:3], axis=2)), axis=2)
            self.ado_est_pose = np.concatenate((self.ado_est_pose, np.reshape([0,0,0,1]*n, (n,1,4))), axis=1)

            ego_state_est_mat = self.est_data['ego_state_est']
            n = len(ego_state_est_mat)
            self.ego_est_pose = np.concatenate((quat_to_rotm(ego_state_est_mat[:, 6:10]),  
                                                np.expand_dims(ego_state_est_mat[:, 0:3], axis=2)), axis=2)
            self.ego_est_pose = np.concatenate((self.ego_est_pose, np.reshape([0,0,0,1]*n, (n,1,4))), axis=1)


        # extract gt
        if 'gt' in log_data:
            self.gt_data = log_data['gt']
            self.t_gt = self.gt_data['time']
            ado_state_gt_mat = self.gt_data['state_gt']
            self.x_gt = ado_state_gt_mat[:, 0]
            self.y_gt = ado_state_gt_mat[:, 1]
            self.z_gt = ado_state_gt_mat[:, 2]
            rpy_mat = quat_to_ang(ado_state_gt_mat[:, 6:10])
            self.roll_gt = rpy_mat[:, 0]
            self.pitch_gt = rpy_mat[:, 1]
            self.yaw_gt = rpy_mat[:, 2]
            
            self.ego_gt_time_pose = self.gt_data['time']  # use same times
            
            n = len(ado_state_gt_mat)
            self.ado_gt_pose = np.concatenate((quat_to_rotm(ado_state_gt_mat[:, 6:10]), 
                                                np.expand_dims(ado_state_gt_mat[:, 0:3], axis=2)), axis=2)
            self.ado_gt_pose = list(np.concatenate((self.ado_gt_pose, np.reshape([0,0,0,1]*n, (n,1,4))), axis=1))

            ego_state_gt_mat = self.gt_data['ego_state_gt']
            n = len(ego_state_gt_mat)
            self.ego_gt_pose = np.concatenate((quat_to_rotm(ego_state_gt_mat[:, 6:10]), 
                                                np.expand_dims(ego_state_gt_mat[:, 0:3], axis=2)), axis=2)
            self.ego_gt_pose = list(np.concatenate((self.ego_gt_pose, np.reshape([0,0,0,1]*n, (n,1,4))), axis=1))


            self.ego_est_time_pose = self.est_data['time']  # use same times
            
            n = len(ado_state_est_mat)
            self.ado_est_pose = np.concatenate((quat_to_rotm(ado_state_est_mat[:, 6:10]), 
                                                np.expand_dims(ado_state_est_mat[:, 0:3], axis=2)), axis=2)
            self.ado_est_pose = np.concatenate((self.ado_est_pose, np.reshape([0,0,0,1]*n, (n,1,4))), axis=1)

            ego_state_est_mat = self.est_data['ego_state_est']
            n = len(ego_state_est_mat)
            self.ego_est_pose = np.concatenate((quat_to_rotm(ego_state_est_mat[:, 6:10]),  
                                                np.expand_dims(ego_state_est_mat[:, 0:3], axis=2)), axis=2)
            self.ego_est_pose = np.concatenate((self.ego_est_pose, np.reshape([0,0,0,1]*n, (n,1,4))), axis=1)

        
        # extract ssp
        if 'ssp' in log_data:
            self.ssp_data = log_data['ssp']
            self.t_ssp = self.ssp_data['time']
            ssp_ado_state_est_mat = self.ssp_data['state_est']
            self.x_ssp = ssp_ado_state_est_mat[:, 0]
            self.y_ssp = ssp_ado_state_est_mat[:, 1]
            self.z_ssp = ssp_ado_state_est_mat[:, 2]
            rpy_mat = quat_to_ang(ssp_ado_state_est_mat[:, 6:10])
            self.roll_ssp = rpy_mat[:, 0]
            self.pitch_ssp = rpy_mat[:, 1]
            self.yaw_ssp = rpy_mat[:, 2]
            
            n = len(ssp_ado_state_est_mat)
            self.ssp_ado_est_pose = np.concatenate((quat_to_rotm(ssp_ado_state_est_mat[:, 6:10]), 
                                                    np.expand_dims(ssp_ado_state_est_mat[:, 0:3], axis=2)), axis=2)
            self.ssp_ado_est_pose = list(np.concatenate((self.ssp_ado_est_pose, np.reshape([0,0,0,1]*n, (n,1,4))), axis=1))


    def quant_eval(self):
        N = len(self.t_est)
        print("Post-processing data now ({} itrs)".format(N))
        corners2D_gt = None

        ###################################

        for i in range(N):
            # extract data in form needed for ssp analysis
            t_est = self.t_est[i]
            tf_w_ado_est = self.ado_est_pose[i]

            tf_w_ego_gt, _ = find_closest_by_time(t_est, self.ego_gt_time_pose, message_list=self.ego_gt_pose)

            tf_w_ego_est, _ = find_closest_by_time(t_est, self.ego_est_time_pose, message_list=self.ego_est_pose)

            tf_w_ado_gt, _ = find_closest_by_time(t_est, self.t_gt, message_list=self.ado_gt_pose)

            tf_w_cam = tf_w_ego_gt @ inv_tf(self.tf_cam_ego)
            tf_cam_w = inv_tf(tf_w_cam)
            tf_cam_ado_est = tf_cam_w @ tf_w_ado_est
            tf_cam_ado_gt = tf_cam_w @ tf_w_ado_gt

            R_pr = tf_cam_ado_est[0:3, 0:3]
            t_pr = tf_cam_ado_est[0:3, 3].reshape((3, 1))
            tf_cam_ado_gt = tf_cam_w @ tf_w_ado_gt
            R_gt = tf_cam_ado_gt[0:3, 0:3]
            t_gt = tf_cam_ado_gt[0:3, 3].reshape((3, 1))
            
            self.raptor_metrics.update_all_metrics(vertices=self.vertices, R_gt=R_gt, t_gt=t_gt, R_pr=R_pr, t_pr=t_pr, K=self.new_camera_matrix)

            ######################################################

        self.raptor_metrics.calc_final_metrics()
        self.raptor_metrics.print_final_metrics()
        print("done with post process!")


    def do_plot(self):
        self.fig, self.axes = plt.subplots(3, 2, clear=True)
        est_line_style = 'r-'
        gt_line_style  = 'b-'
        ssp_line_style = 'm-'
        ang_type = 'rad'
        if self.b_degrees:
            self.roll_gt *= 180/np.pi
            self.roll_est *= 180/np.pi
            self.pitch_gt *= 180/np.pi
            self.pitch_est *= 180/np.pi
            self.yaw_gt *= 180/np.pi
            self.yaw_est *= 180/np.pi
            if self.ssp_data is not None:
                self.roll_ssp *= 180/np.pi
                self.pitch_ssp *= 180/np.pi
                self.yaw_ssp *= 180/np.pi
            ang_type = 'deg'

        self.x_gt_plt, = self.axes[0,0].plot(self.t_gt, self.x_gt, gt_line_style)
        self.x_est_plt, = self.axes[0,0].plot(self.t_est, self.x_est, est_line_style)
        self.axes[0, 0].set_ylabel("x [m]")

        self.y_gt_plt, = self.axes[1,0].plot(self.t_gt, self.y_gt, gt_line_style)
        self.y_est_plt, = self.axes[1,0].plot(self.t_est, self.y_est, est_line_style)
        self.axes[1, 0].set_ylabel("y [m]")

        self.z_gt_plt, = self.axes[2,0].plot(self.t_gt, self.z_gt, gt_line_style)
        self.z_est_plt, = self.axes[2,0].plot(self.t_est, self.z_est, est_line_style)
        self.axes[2, 0].set_ylabel("z [m]")

        self.roll_gt_plt, = self.axes[0,1].plot(self.t_gt, self.roll_gt, gt_line_style)
        self.roll_est_plt, = self.axes[0,1].plot(self.t_est, self.roll_est, est_line_style)
        self.axes[0, 1].set_ylabel("roll [{}]".format(ang_type))

        self.pitch_gt_plt, = self.axes[1,1].plot(self.t_gt, self.pitch_gt, gt_line_style)
        self.pitch_est_plt, = self.axes[1,1].plot(self.t_est, self.pitch_est, est_line_style)
        self.axes[1, 1].set_ylabel("pitch [{}]".format(ang_type))

        self.yaw_gt_plt, = self.axes[2,1].plot(self.t_gt, self.yaw_gt, gt_line_style)
        self.yaw_est_plt, = self.axes[2,1].plot(self.t_est, self.yaw_est, est_line_style)
        self.axes[2, 1].set_ylabel("yaw [{}]".format(ang_type))

        if self.ssp_data is not None:
            self.x_ssp_plt, = self.axes[0,0].plot(self.t_ssp, self.x_ssp, ssp_line_style)
            self.y_ssp_plt, = self.axes[1,0].plot(self.t_ssp, self.y_ssp, ssp_line_style)
            self.z_ssp_plt, = self.axes[2,0].plot(self.t_ssp, self.z_ssp, ssp_line_style)
            self.roll_ssp_plt, = self.axes[0,1].plot(self.t_ssp, self.roll_ssp, ssp_line_style)
            self.pitch_ssp_plt, = self.axes[1,1].plot(self.t_ssp, self.pitch_ssp, ssp_line_style)
            self.yaw_ssp_plt, = self.axes[2,1].plot(self.t_ssp, self.yaw_ssp, ssp_line_style)


        for ax in np.reshape(self.axes, (self.axes.size)):
            ax.set_xlim([0, self.tf])
            ax.set_xlabel("time (s)")
            yl1, yl2 = ax.get_ylim()
            for ts, tf in self.detect_times:
                if np.isnan(tf) or tf - ts < 0.1: # detect mode happened just once - draw line
                    ax.plot([ts, ts], [-1e4, 1e4], linestyle='-', color="#d62728", linewidth=0.5) # using yl1 and yl2 for the line plot doesnt span the full range
                else: # detect mode happened for a span - draw rect
                    ax.axvspan(ts, tf, facecolor='#d62728', alpha=0.5)  # red: #d62728, blue: 1f77b4, green: #2ca02c
            ax.set_ylim([yl1, yl2])

        if self.ssp_data is not None:
            plt.suptitle("MSL-RAPTOR Results (Blue - GT, Red - Est, Magenta - SSP)")
        else:
            plt.suptitle("MSL-RAPTOR Results (Blue - GT, Red - Est)")
        plt.show(block=False)


if __name__ == '__main__':
    try:
        if len(sys.argv) == 1:
            raise RuntimeError("not enough arguments, must pass in the logfile name or source rosbag name (w/ or w/o .bag)")
        elif len(sys.argv) > 2:
            raise RuntimeError("too many arguments, only pass in the rosbag name or source rosbag name (w/ or w/o .bag)")
        np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
        program = result_analyser(log_identifier=sys.argv[1])
        input("\nPress enter to close program\n")
        
    except:
        import traceback
        traceback.print_exc()

