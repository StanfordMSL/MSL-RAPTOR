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
from raptor_logger import *
from pose_metrics import *

class rosbags_to_logs:
    """
    This class takes in the rosbag that is output from mslraptor and processes it into our custom log format 
    this enables us to unify output with the baseline method)
    The code currently also runs the quantitative metric analysis in the processes, but this is optional and will be done 
    again in the result_analyser. 
    """
    def __init__(self, rb_name=None, ego_quad_ns="/quad7", ado_quad_ns="/quad4"):
        
        us_split = rb_name.split("_")
        if rb_name[-4:] == '.bag' or "_".join(us_split[0:3]) == 'msl_raptor_output':
            # This means rosbag name is one that was post-processed
            if len(rb_name) > 4 and rb_name[-4:] == ".bag":
                self.rb_name = rb_name
            else:
                self.rb_name = rb_name + ".bag"
        elif len(rb_name) > 4 and "_".join(us_split[0:4]) == 'rosbag_for_post_process':
            # we assume this is the rosbag that fed into raptor
            rb_name = "msl_raptor_output_from_bag_rosbag_for_post_process_" + us_split[4]
            if rb_name[-4:] == ".bag":
                self.rb_name = rb_name
            else:
                self.rb_name = rb_name + ".bag"
        else:
            raise RuntimeError("We do not recognize bag file! {} not understood".format(rb_name))
        
        self.rosbag_in_dir = "/mounted_folder/raptor_processed_bags"
        self.log_out_dir = "/mounted_folder/raptor_logs"
        try:
            self.bag = rosbag.Bag(self.rosbag_in_dir + '/' + self.rb_name, 'r')
        except Exception as e:
            raise RuntimeError("Unable to Process Rosbag!!\n{}".format(e))

        self.ado_gt_topic    = ado_quad_ns + '/mavros/vision_pose/pose'
        self.ado_est_topic   = ego_quad_ns + '/msl_raptor_state'  # ego_quad_ns since it is ego_quad's estimate of the ado quad
        self.bb_data_topic   = ego_quad_ns + '/bb_data'  # ego_quad_ns since it is ego_quad's estimate of the bounding box
        self.ego_gt_topic    = ego_quad_ns + '/mavros/vision_pose/pose'
        self.ego_est_topic   = ego_quad_ns + '/mavros/local_position/pose'
        self.cam_info_topic  = ego_quad_ns + '/camera/camera_info'
        self.topic_func_dict = {self.ado_gt_topic : self.parse_ado_gt_msg, 
                                self.ado_est_topic : self.parse_ado_est_msg, 
                                self.bb_data_topic : self.parse_bb_msg,
                                self.ego_gt_topic : self.parse_ego_gt_msg, 
                                self.ego_est_topic : self.parse_ego_est_msg,
                                self.cam_info_topic: self.parse_camera_info_msg}

        self.b_degrees = True  # use degrees or radians

        self.fig = None
        self.axes = None
        self.name = 'mslquad'
        self.t0 = -1
        self.tf = -1
        self.t_est = []
        self.t_gt = []

        self.ego_gt_time_pose = []
        self.ego_gt_pose = []
        self.ego_est_pose = []
        self.ego_est_time_pose = []
        self.ado_gt_pose = []
        self.ado_est_pose = []
        self.ado_est_state = []

        self.DETECT = 1
        self.TRACK = 2
        self.FAKED_BB = 3
        self.IGNORE = 4
        self.detect_time = []
        self.detect_mode = []
        
        self.detect_times = []
        self.detect_end = None
        self.abb_list = []
        self.abb_time_list = []

        # Create camera (camera extrinsics from quad7.param in the msl_raptor project):
        self.tf_cam_ego = np.eye(4)
        self.tf_cam_ego[0:3, 3] = np.asarray([0.01504337, -0.06380886, -0.13854437])
        self.tf_cam_ego[0:3, 0:3] = np.reshape([-6.82621737e-04, -9.99890488e-01, -1.47832690e-02, 3.50423970e-02,  1.47502748e-02, -9.99276969e-01, 9.99385593e-01, -1.20016936e-03,  3.50284906e-02], (3, 3))
        
        # Correct Rotation w/ Manual Calibration
        Angle_x = 8./180. 
        Angle_y = 8./180.
        Angle_z = 0./180. 
        R_deltax = np.array([[ 1.             , 0.             , 0.              ],
                             [ 0.             , np.cos(Angle_x),-np.sin(Angle_x) ],
                             [ 0.             , np.sin(Angle_x), np.cos(Angle_x) ]])
        R_deltay = np.array([[ np.cos(Angle_y), 0.             , np.sin(Angle_y) ],
                             [ 0.             , 1.             , 0               ],
                             [-np.sin(Angle_y), 0.             , np.cos(Angle_y) ]])
        R_deltaz = np.array([[ np.cos(Angle_z),-np.sin(Angle_z), 0.              ],
                             [ np.sin(Angle_z), np.cos(Angle_z), 0.              ],
                             [ 0.             , 0.             , 1.              ]])
        R_delta = np.dot(R_deltax, np.dot(R_deltay, R_deltaz))
        self.tf_cam_ego[0:3,0:3] = np.matmul(R_delta, self.tf_cam_ego[0:3,0:3])
        self.K = None
        self.dist_coefs = None
        self.new_camera_matrix = None
        #########################################################################################
        
        est_log_name   = self.log_out_dir + "/log_" + rb_name[:-4].split("_")[-1] + "_EST.log"
        gt_log_name    = self.log_out_dir + "/log_" + rb_name[:-4].split("_")[-1] + "_GT.log"
        param_log_name = self.log_out_dir + "/log_" + rb_name[:-4].split("_")[-1] + "_PARAM.log"
        self.logger = raptor_logger(source="MSLRAPTOR", mode="write", est_fn=est_log_name, gt_fn=gt_log_name, param_fn=param_log_name)
        self.process_rb()

        self.diam = 0.311
        self.box_length = 0.27
        self.box_width = 0.27
        self.box_height = 0.13

        self.raptor_metrics = pose_metric_tracker(px_thresh=5, prct_thresh=10, trans_thresh=0.05, ang_thresh=5, name=self.name, diam=self.diam)
        
        self.process_rosbag()
        self.logger.close_files()


    def process_rosbag(self):
        N = len(self.t_est)
        print("Post-processing data now ({} itrs)".format(N))
        vertices = np.array([[ self.box_length/2, self.box_width/2, self.box_height/2, 1.],
                             [ self.box_length/2, self.box_width/2,-self.box_height/2, 1.],
                             [ self.box_length/2,-self.box_width/2,-self.box_height/2, 1.],
                             [ self.box_length/2,-self.box_width/2, self.box_height/2, 1.],
                             [-self.box_length/2,-self.box_width/2, self.box_height/2, 1.],
                             [-self.box_length/2,-self.box_width/2,-self.box_height/2, 1.],
                             [-self.box_length/2, self.box_width/2,-self.box_height/2, 1.],
                             [-self.box_length/2, self.box_width/2, self.box_height/2, 1.]]).T

        # Write params to log file ########
        log_data = {}
        if self.new_camera_matrix is not None:
            log_data['K'] = np.array([self.new_camera_matrix[0, 0], self.new_camera_matrix[1, 1], self.new_camera_matrix[0, 2], self.new_camera_matrix[1, 2]])
        else:
            log_data['K'] = np.array([self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]])
        log_data['3d_bb_dims'] = np.array([self.box_length, self.box_width, self.box_height, self.diam])
        log_data['tf_cam_ego'] = np.reshape(self.tf_cam_ego, (16,))
        self.logger.write_data_to_log(log_data, mode='prms')
        ###################################

        for i in range(N):
            # extract data in form for logging
            t_est = self.t_est[i]
            tf_w_ado_est = pose_to_tf(self.ado_est_pose[i])

            pose_msg, _ = find_closest_by_time(t_est, self.ego_gt_time_pose, message_list=self.ego_gt_pose)
            tf_w_ego_gt = pose_to_tf(pose_msg)

            pose_msg, _ = find_closest_by_time(t_est, self.ego_est_time_pose, message_list=self.ego_est_pose)
            tf_w_ego_est = pose_to_tf(pose_msg)

            pose_msg, _ = find_closest_by_time(t_est, self.t_gt, message_list=self.ado_gt_pose)
            tf_w_ado_gt = pose_to_tf(pose_msg)

            tf_w_cam = tf_w_ego_gt @ inv_tf(self.tf_cam_ego)
            tf_cam_w = inv_tf(tf_w_cam)
            tf_cam_ado_est = tf_cam_w @ tf_w_ado_est
            tf_cam_ado_gt = tf_cam_w @ tf_w_ado_gt

            R_pr = tf_cam_ado_est[0:3, 0:3]
            t_pr = tf_cam_ado_est[0:3, 3].reshape((3, 1))
            tf_cam_ado_gt = tf_cam_w @ tf_w_ado_gt
            R_gt = tf_cam_ado_gt[0:3, 0:3]
            t_gt = tf_cam_ado_gt[0:3, 3].reshape((3, 1))
            
            ######################################################
            
            if self.raptor_metrics is not None:
                self.raptor_metrics.update_all_metrics(vertices=vertices, R_gt=R_gt, t_gt=t_gt, R_pr=R_pr, t_pr=t_pr, K=self.new_camera_matrix)

            # Write data to log file #############################
            (abb, im_seg_mode), _ = find_closest_by_time(t_est, self.abb_time_list, message_list=self.abb_list)
            log_data['time'] = t_est
            log_data['state_est'] = tf_to_state_vec(tf_w_ado_est)
            log_data['state_gt'] = tf_to_state_vec(tf_w_ado_gt)
            log_data['ego_state_est'] = tf_to_state_vec(tf_w_ego_est)
            log_data['ego_state_gt'] = tf_to_state_vec(tf_w_ego_gt)
            corners3D_pr = (tf_w_ado_est @ vertices)[0:3,:]
            corners3D_gt = (tf_w_ado_gt @ vertices)[0:3,:]
            log_data['corners_3d_est'] = np.reshape(corners3D_pr, (corners3D_pr.size,))
            log_data['corners_3d_gt'] = np.reshape(corners3D_gt, (corners3D_gt.size,))
            log_data['proj_corners_est'] = np.reshape(self.raptor_metrics.proj_2d_pr.T, (self.raptor_metrics.proj_2d_pr.size,))
            log_data['proj_corners_gt'] = np.reshape(self.raptor_metrics.proj_2d_gt.T, (self.raptor_metrics.proj_2d_gt.size,))
            log_data['abb'] = abb
            log_data['im_seg_mode'] = im_seg_mode
            self.logger.write_data_to_log(log_data, mode='est')
            self.logger.write_data_to_log(log_data, mode='gt')
            ######################################################

        if self.raptor_metrics is not None:
            self.raptor_metrics.calc_final_metrics()
            self.raptor_metrics.print_final_metrics()
        print("done processing rosbag into logs!")


    def process_rb(self):
        print("Processing {}".format(self.rb_name))
        for topic, msg, t in self.bag.read_messages( topics=list(self.topic_func_dict.keys()) ):
            self.topic_func_dict[topic](msg)
        self.t_est = np.asarray(self.t_est)
        self.t0 = np.min(self.t_est)
        self.tf = np.max(self.t_est) - self.t0
        self.t_est -= self.t0
        self.t_gt = np.asarray(self.t_gt) - self.t0
        self.detect_time = np.asarray(self.detect_time) - self.t0
        self.detect_times = np.asarray(self.detect_times) - self.t0


    def parse_camera_info_msg(self, msg, t=None):
        if self.K is None:
            camera_info = msg
            self.K = np.reshape(camera_info.K, (3, 3))
            if len(camera_info.D) == 5:
                self.dist_coefs = np.reshape(camera_info.D, (5,))
                self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist_coefs, (camera_info.width, camera_info.height), 0, (camera_info.width, camera_info.height))
            else:
                self.dist_coefs = None
                self.new_camera_matrix = self.K
    
    def parse_ado_est_msg(self, msg, t=None):
        """
        record estimated poses from msl-raptor
        """
        tracked_obs = msg.tracked_objects
        if len(tracked_obs) == 0:
            return
        to = tracked_obs[0]  # assumes 1 object for now
        if t is None:
            self.t_est.append(to.pose.header.stamp.to_sec())
        else:
            self.t_est.append(t)

        self.ado_est_pose.append(to.pose.pose)
        self.ado_est_state.append(to.pose.pose)


    def parse_ado_gt_msg(self, msg, t=None):
        """
        record optitrack poses of tracked quad
        """
        if t is None:
            self.t_gt.append(msg.header.stamp.to_sec())
        else:
            self.t_gt.append(t)

        self.ado_gt_pose.append(msg.pose)

        
    def parse_ego_gt_msg(self, msg, t=None):
        """
        record optitrack poses of tracked quad
        """
        self.ego_gt_pose.append(msg.pose)
        self.ego_gt_time_pose.append(msg.header.stamp.to_sec())

        
    def parse_ego_est_msg(self, msg, t=None):
        """
        record optitrack poses of tracked quad
        """
        self.ego_est_pose.append(msg.pose)
        self.ego_est_time_pose.append(msg.header.stamp.to_sec())


    def parse_bb_msg(self, msg, t=None):
        """
        record times of detect
        note message is custom MSL-RAPTOR angled bounding box
        """
        msg = msg.boxes[0]
        t = msg.header.stamp.to_sec()
        if msg.im_seg_mode == self.DETECT:
            self.detect_time.append(t)

        self.abb_list.append(([msg.x, msg.y, msg.width, msg.height, msg.angle*180./np.pi], msg.im_seg_mode))
        self.abb_time_list.append(t)
        ######
        eps = 0.1 # min width of line
        if msg.im_seg_mode == self.DETECT:  # we are detecting now
            if not self.detect_times:  # first run - init list
                self.detect_times = [[t]]
                self.detect_end = np.nan
            elif len(self.detect_times[-1]) == 2: # we are starting a new run
                self.detect_times.append([t])
                self.detect_end = np.nan
            else: # len(self.detect_times[-1]) = 1: # we are currently still on a streak of detects
                self.detect_end = t
        else: # not detecting
            if not self.detect_times or len(self.detect_times[-1]) == 2: # we are still not detecting (we were not Detecting previously)
                pass
            else: # self.detect_times[-1][1]: # we were just tracking
                self.detect_times[-1].append(self.detect_end)
                self.detect_end = np.nan

        self.detect_mode.append(msg.im_seg_mode)


if __name__ == '__main__':
    try:
        if len(sys.argv) == 1:
            raise RuntimeError("not enough arguments, must pass in the rosbag name (w/ or w/o .bag)")
        elif len(sys.argv) > 2:
            raise RuntimeError("too many arguments, only pass in the rosbag name (w/ or w/o .bag)")
        np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
        program = rosbags_to_logs(rb_name=sys.argv[1])
        
    except:
        import traceback
        traceback.print_exc()

