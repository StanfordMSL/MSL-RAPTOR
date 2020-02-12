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
from ssp_utils import *

class result_analyser:

    def __init__(self, rb_name=None, ego_quad_ns="/quad7", ado_quad_ns="/quad4"):
        if rb_name is not None:
            if len(rb_name) > 4 and rb_name[-4:] == ".bag":
                self.rb_name = rb_name
            else:
                self.rb_name = rb_name + ".bag"
        else:
            raise RuntimeError("rb_name = None - need to provide rosbag name (with or without .bag)")

        try:
            self.bag = rosbag.Bag(self.rb_name, 'r')
        except Exception as e:
            raise RuntimeError("Unable to Process Rosbag!!\n{}".format(e))

        self.ado_gt_topic = ado_quad_ns + '/mavros/vision_pose/pose'
        self.ado_est_topic = ego_quad_ns + '/msl_raptor_state'  # ego_quad_ns since it is ego_quad's estimate of the ado quad
        self.bb_data_topic = ego_quad_ns + '/bb_data'  # ego_quad_ns since it is ego_quad's estimate of the bounding box
        self.ego_gt_topic = ego_quad_ns + '/mavros/vision_pose/pose'
        self.cam_info_topic = ego_quad_ns + '/camera/camera_info'
        self.topic_func_dict = {self.ado_gt_topic : self.parse_ado_gt_msg, 
                                self.ado_est_topic : self.parse_ado_est_msg, 
                                self.bb_data_topic : self.parse_bb_msg,
                                self.ego_gt_topic : self.parse_ego_gt_msg, 
                                self.cam_info_topic: self.parse_camera_info_msg}

        self.b_degrees = True  # use degrees or radians

        self.fig = None
        self.axes = None
        self.name = 'mslquad'
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

        self.process_rb()
        self.do_plot()
        self.quat_eval()


    def quat_eval(self):
        N = len(self.t_est)
        print("Post-processing data now ({} itrs)".format(N))

        # To save
        trans_dist = 0.0
        angle_dist = 0.0
        pixel_dist = 0.0
        testing_samples = 0.0
        testing_error_trans = 0.0
        testing_error_angle = 0.0
        testing_error_pixel = 0.0
        errs_2d             = []
        errs_3d             = []
        errs_trans          = []
        errs_angle          = []
        errs_corner2D       = []
        preds_trans         = []
        preds_rot           = []
        preds_corners2D     = []
        gts_trans           = []
        gts_rot             = []
        gts_corners2D       = []
        corners2D_gt = None

        diam = 0.311
        box_length = 0.27
        box_width = 0.27
        box_height = 0.13
        vertices = np.array([[ box_length/2, box_width/2, box_height/2, 1.],
                            [ box_length/2, box_width/2,-box_height/2, 1.],
                            [ box_length/2,-box_width/2,-box_height/2, 1.],
                            [ box_length/2,-box_width/2, box_height/2, 1.],
                            [-box_length/2,-box_width/2, box_height/2, 1.],
                            [-box_length/2,-box_width/2,-box_height/2, 1.],
                            [-box_length/2, box_width/2,-box_height/2, 1.],
                            [-box_length/2, box_width/2, box_height/2, 1.]]).T
        # corners3D = get_3D_corners(vertices)
        
        for i in range(N):
            # extract data in form needed for ssp analysis
            t_est = self.t_est[i]
            tf_w_ado_est = pose_to_tf(self.ado_est_pose[i])

            ego_msg, _ = find_closest_by_time(t_est, self.ego_gt_time_pose, message_list=self.ego_gt_pose)
            tf_w_ego_gt = pose_to_tf(ego_msg)

            ado_msg, _ = find_closest_by_time(t_est, self.t_gt, message_list=self.ado_gt_pose)
            tf_w_ado_gt = pose_to_tf(ego_msg)

            tf_w_cam = tf_w_ego_gt @ invert_tf(self.tf_cam_ego)
            tf_cam_w = invert_tf(tf_w_cam)
            tf_cam_ado_est = tf_cam_w @ tf_w_ado_est
            tf_cam_ado_gt = tf_cam_w @ tf_w_ado_gt

            R_pr = tf_cam_ado_est[0:3, 0:3]
            t_pr = tf_cam_ado_est[0:3, 3].reshape((3, 1))
            tf_cam_ado_gt = tf_cam_w @ tf_w_ado_gt
            R_gt = tf_cam_ado_gt[0:3, 0:3]
            t_gt = tf_cam_ado_gt[0:3, 3].reshape((3, 1))
            Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
            Rt_pr = np.concatenate((R_pr, t_pr), axis=1)

            corners2D_pr = compute_projection(np.hstack((np.reshape([0,0,0,1], (4,1)), vertices)), Rt_pr, self.new_camera_matrix).T
            corners2D_gt = compute_projection(np.hstack((np.reshape([0,0,0,1], (4,1)), vertices)), Rt_gt, self.new_camera_matrix).T
            
            ######################################################

            # Compute translation error
            trans_dist = np.sqrt(np.sum(np.square(t_gt - t_pr)))
            errs_trans.append(trans_dist)
            
            # Compute angle error
            angle_dist = calcAngularDistance(R_gt, R_pr)
            errs_angle.append(angle_dist)
            
            # Compute pixel error
            Rt_gt        = np.concatenate((R_gt, t_gt), axis=1)
            Rt_pr        = np.concatenate((R_pr, t_pr), axis=1)
            proj_2d_gt   = compute_projection(vertices, Rt_gt, self.new_camera_matrix)
            proj_2d_pred = compute_projection(vertices, Rt_pr, self.new_camera_matrix) 
            norm         = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
            pixel_dist   = np.mean(norm)
            errs_2d.append(pixel_dist)

            # Compute corner prediction error
            corners2D_gt = compute_projection(np.hstack((np.reshape([0,0,0,1], (4,1)), vertices)), Rt_gt, self.new_camera_matrix).T
            corner_norm = np.linalg.norm(corners2D_gt - corners2D_pr, axis=1)
            corner_dist = np.mean(corner_norm)
            errs_corner2D.append(corner_dist)

            # Compute 3D distances
            transform_3d_gt   = compute_transformation(vertices, Rt_gt) 
            transform_3d_pred = compute_transformation(vertices, Rt_pr)  
            norm3d            = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
            vertex_dist       = np.mean(norm3d)
            errs_3d.append(vertex_dist)

            # Sum errors
            testing_error_trans  += trans_dist
            testing_error_angle  += angle_dist
            testing_error_pixel  += pixel_dist
            testing_samples      += 1
            # if b_save_bb_imgs:
            #     draw_2d_proj_of_3D_bounding_box(img, corners2D_pr, corners2D_gt=corners2D_gt, epoch=None, batch_idx=None, detect_num=i, im_save_dir=bb_im_path)

        # Compute 2D projection error, 6D pose error, 5cm5degree error
        px_threshold = 5 # 5 pixel threshold for 2D reprojection error is standard in recent sota 6D object pose estimation works 
        eps          = 1e-5
        acc          = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
        acc5cm5deg   = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (len(errs_trans)+eps)
        acc3d10      = len(np.where(np.array(errs_3d) <= diam * 0.1)[0]) * 100. / (len(errs_3d)+eps)
        acc5cm5deg   = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (len(errs_trans)+eps)
        corner_acc   = len(np.where(np.array(errs_corner2D) <= px_threshold)[0]) * 100. / (len(errs_corner2D)+eps)
        mean_err_2d  = np.mean(errs_2d)
        mean_corner_err_2d = np.mean(errs_corner2D)
        nts = float(testing_samples)

        # Print test statistics
        logging('\nResults of {}'.format(self.name))
        logging('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))
        logging('   Acc using 10% threshold - {} vx 3D Transformation = {:.2f}%'.format(diam * 0.1, acc3d10))
        logging('   Acc using 5 cm 5 degree metric = {:.2f}%'.format(acc5cm5deg))
        logging("   Mean 2D pixel error is %f, Mean vertex error is %f, mean corner error is %f" % (mean_err_2d, np.mean(errs_3d), mean_corner_err_2d))
        logging('   Translation error: %f m, angle error: %f degree, pixel error: % f pix' % (testing_error_trans/nts, testing_error_angle/nts, testing_error_pixel/nts) )
        pdb.set_trace()

        print("done with post process!")


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


    def do_plot(self):
        self.fig, self.axes = plt.subplots(3, 2, clear=True)
        est_line_style = 'r-'
        gt_line_style = 'b-'
        ang_type = 'rad'
        if self.b_degrees:
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

        for ax in np.reshape(self.axes, (self.axes.size)):
            ax.set_xlim([0, self.tf])
            ax.set_xlabel("time (s)")
            yl1, yl2 = ax.get_ylim()
            for ts, tf in self.detect_times:
                if np.isnan(tf) or tf - ts < 0.1: # detect mode happened just once - draw line
                    # ax.axvline(ts, 'r-') # FOR SOME REASON THIS DOES NOT WORK!!
                    ax.plot([ts, ts], [-1e4, 1e4], linestyle='-', color="#d62728", linewidth=0.5) # using yl1 and yl2 for the line plot doesnt span the full range
                else: # detect mode happened for a span - draw rect
                    ax.axvspan(ts, tf, facecolor='#d62728', alpha=0.5)  # red: #d62728, blue: 1f77b4, green: #2ca02c
            ax.set_ylim([yl1, yl2])

        plt.suptitle("MSL-RAPTOR Results (Blue - Est, Red - GT, Green - Front End Detect Mode)")
        plt.show(block=False)
       

    def parse_camera_info_msg(self, msg, t=None):
        if self.K is None:
            camera_info = msg
            self.K = np.reshape(camera_info.K, (3, 3))
            self.dist_coefs = np.reshape(camera_info.D, (5,))
            self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist_coefs, (camera_info.width, camera_info.height), 0, (camera_info.width, camera_info.height))
        
    
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

        my_state = pose_to_state_vec(to.pose.pose)
        rpy = quat_to_ang(np.reshape(my_state[6:10], (1,4)), b_degrees=self.b_degrees)[0]
        self.ado_est_pose.append(to.pose.pose)
        self.ado_est_state.append(to.pose.pose)
        self.x_est.append(my_state[0])
        self.y_est.append(my_state[1])
        self.z_est.append(my_state[2])
        self.roll_est.append(rpy[0])
        self.pitch_est.append(rpy[1])
        self.yaw_est.append(rpy[2])


    def parse_ado_gt_msg(self, msg, t=None):
        """
        record optitrack poses of tracked quad
        """
        if t is None:
            self.t_gt.append(msg.header.stamp.to_sec())
        else:
            self.t_gt.append(t)

        my_state = pose_to_state_vec(msg.pose)
        rpy = quat_to_ang(np.reshape(my_state[6:10], (1,4)))[0]
        self.ado_gt_pose.append(msg.pose)
        self.x_gt.append(my_state[0])
        self.y_gt.append(my_state[1])
        self.z_gt.append(my_state[2])
        self.roll_gt.append(rpy[0])
        self.pitch_gt.append(rpy[1])
        self.yaw_gt.append(rpy[2])

        
    def parse_ego_gt_msg(self, msg, t=None):
        """
        record optitrack poses of tracked quad
        """
        self.ego_gt_pose.append(msg.pose)
        self.ego_gt_time_pose.append(msg.header.stamp.to_sec())


    def parse_bb_msg(self, msg, t=None):
        """
        record times of detect
        note message is custom MSL-RAPTOR angled bounding box
        """
        msg = msg.boxes[0]
        t = msg.header.stamp.to_sec()
        if msg.im_seg_mode == self.DETECT:
            self.detect_time.append(t)

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
        program = result_analyser(rb_name=sys.argv[1])
        input("\nPress enter to close program\n")
        
    except:
        import traceback
        traceback.print_exc()

