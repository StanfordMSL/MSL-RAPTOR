#!/usr/bin/env python3
# IMPORTS
# system
import sys, os, time
from copy import copy
from collections import defaultdict
import yaml
import pdb
# math
import numpy as np
from scipy.optimize import linear_sum_assignment as scipy_hung_alg
from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt
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
from ukf_utils import *
from raptor_logger import *
from pose_metrics import *
from viz_utils import *

class rosbags_to_logs:
    """
    This class takes in the rosbag that is output from mslraptor and processes it into our custom log format 
    this enables us to unify output with the baseline method)
    The code currently also runs the quantitative metric analysis in the processes, but this is optional and will be done 
    again in the result_analyser. 
    """
    def __init__(self, rb_name=None, data_source='raptor', ego_quad_ns="/quad7", ego_yaml="quad7", ado_yaml="all_obj", b_save_3dbb_imgs=False):
        # Parse rb_name
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
        self.log_out_dir = "/mounted_folder/" + data_source.lower() + "_logs"
        makedirs(self.log_out_dir)

        try:
            self.bag = rosbag.Bag(self.rosbag_in_dir + '/' + self.rb_name, 'r')
        except Exception as e:
            raise RuntimeError("Unable to Process Rosbag!!\n{}".format(e))

        self.bags_and_cut_times = {"msl_raptor_output_from_bag_rosbag_for_post_process_2019-12-18-02-10-28.bag" : 1576663867,
                                   "msl_raptor_output_from_bag_rosbag_for_post_process_TX2_2019-12-18-02-10-28.bag" : 1576663867}


        self.bb_data_topic   = ego_quad_ns + '/bb_data'  # ego_quad_ns since it is ego_quad's estimate of the bounding box
        self.ego_gt_topic    = ego_quad_ns + '/mavros/vision_pose/pose'
        self.ego_est_topic   = ego_quad_ns + '/mavros/local_position/pose'
        self.cam_info_topic  = ego_quad_ns + '/camera/camera_info'
        self.topic_func_dict = {self.bb_data_topic : self.parse_bb_msg,
                                self.ego_gt_topic  : self.parse_ego_gt_msg, 
                                self.ego_est_topic : self.parse_ego_est_msg,
                                self.cam_info_topic: self.parse_camera_info_msg}
        self.camera_topic = ego_quad_ns + '/camera/image_raw'
        self.ego_quad_ns = ego_quad_ns
        self.b_save_3dbb_imgs = b_save_3dbb_imgs
        if self.b_save_3dbb_imgs:
            self.bridge = CvBridge()
            self.processed_image_dict = {} 
            self.topic_func_dict[self.camera_topic] = self.parse_camera_img_msg
        self.color_list = [(255, 0, 255),   # 0 magenta
                           (0, 255, 0),     # 1 green
                           (255, 0, 0),     # 2 blue
                           (255, 255, 0),   # 3 cyan
                           (0, 0, 255),     # 4 red
                           (0, 255, 255),   # 5 yellow
                           (255, 255, 255), # 7 white
                           (0, 0, 0),       # 6 black
                           (125, 125, 125)] # 8 grey
        self.ado_name_to_color = {}
        self.bb_linewidth = 1
        self.img_time_buffer = []
        self.img_msg_buffer = []

        # Read yaml file to get object params
        self.ado_names = set()  # names actually in our rosbag
        self.ado_names_all = []  # all names we could possibly recognize
        self.bb_3d_dict_all = defaultdict(list)  # turns the name into the len, width, height, and diam of the object
        self.tf_cam_ego = None
        self.class_str_to_name_dict = defaultdict(list)
        self.ado_name_to_class = {} # pulled from the all_obs yaml file
        self.b_enforce_0_dict = {}
        self.fixed_vals = {}
        self.read_yaml(ego_yaml=ego_yaml, ado_yaml=ado_yaml)
        ####################################

        self.fig = None
        self.axes = None
        self.name = 'mslquad'
        self.t0 = -1
        self.tf = -1
        self.t_est = set()
        self.t_gt = defaultdict(list)

        self.ego_gt_time_pose = []
        self.ego_gt_pose = []
        self.ego_est_pose = []
        self.ego_est_time_pose = []
        self.ado_gt_pose = defaultdict(list)

        self.ado_est_pose_BY_TIME_BY_CLASS = defaultdict(dict)
        self.ado_est_state = defaultdict(list)

        self.DETECT = 1
        self.TRACK = 2
        self.FAKED_BB = 3
        self.IGNORE = 4
        self.detect_time = []
        self.detect_mode = []
        
        self.abb_list = defaultdict(list)
        self.abb_time_list = defaultdict(list)

        self.K = None
        self.dist_coefs = None
        self.new_camera_matrix = None
        self.camera = None
        #########################################################################################
        self.b_plot_gt_overlay = True
        if self.b_plot_gt_overlay:
            objects_sizes_yaml  = "/root/msl_raptor_ws/src/msl_raptor/params/all_obs.yaml"
            objects_used_path_and_file  = "/root/msl_raptor_ws/src/msl_raptor/params/objects_used/objects_used.txt"
            classes_names_file = "/root/msl_raptor_ws/src/msl_raptor/params/classes.names"
            category_params = load_category_params() # Returns dict of params per class name
            # bb_3d, obj_width, obj_height, classes_names, classes_ids, objects_names_per_class, connected_inds = \
            self.info_for_gt_overlay = get_object_sizes_from_yaml(objects_sizes_yaml, objects_used_path_and_file, classes_names_file, category_params)  # Parse objects used and associated configurations

        self.process_rb()
        
        base_path = self.log_out_dir + "/log_" + self.rb_name[:-4].split("_")[-1] 
        self.logger = RaptorLogger(mode="write", names=self.ado_names, base_path=base_path)

        self.raptor_metrics = PoseMetricTracker(px_thresh=5, prct_thresh=10, trans_thresh=0.05, ang_thresh=5, names=self.ado_names, bb_3d_dict=self.bb_3d_dict_all)
        
        self.convert_rosbag_info_to_log()
        self.logger.close_files()


    def find_closest_pose_est_by_class_and_time(self, tf_w_ado_gt_dict, candidate_poses, candidate_object_names, close_cutoff=0.5):
        """
        first narrow the field by distance, then once things are "close" start using angle also
        """
        best_pose = None
        min_trans_diff = 1e10
        min_rot_diff = np.pi
        R_gt = tf_w_ado_gt[0:3, 0:3]
        t_gt = tf_w_ado_gt[0:3, 3]
        for pose in candidate_poses:
            tf_pr = pose_to_tf(pose)
            R_pr = tf_pr[0:3, 0:3]
            t_pr = tf_pr[0:3, 3]
            dtran = la.norm(t_pr - t_gt)
            if dtran > 2*close_cutoff:
                continue

            dang = calcAngularDistance(R_gt, R_pr)
            if min_trans_diff > 1e5: # accept regardless of rotation, but only if we havent already accepted another option
                min_trans_diff = dtran
                min_rot_diff = dang
                best_pose = tf_pr
            elif min_trans_diff > dtran and min_rot_diff*1.2 > dang:  # accept if we are closer and angle isnt much worse
                min_trans_diff = dtran
                min_rot_diff = dang
                best_pose = tf_pr
        return best_pose


    def convert_rosbag_info_to_log(self):
        # Write params to log file ########
        param_data = {}
        if self.new_camera_matrix is not None:
            param_data['K'] = np.array([self.new_camera_matrix[0, 0], self.new_camera_matrix[1, 1], self.new_camera_matrix[0, 2], self.new_camera_matrix[1, 2]])
        else:
            param_data['K'] = np.array([self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]])
        bb_dim_arr = np.zeros((4*len(self.ado_names)))
        for i, n in enumerate(self.ado_names):
            bb_dim_arr[4*i:4*i+4] = self.bb_3d_dict_all[n]
        param_data['3d_bb_dims'] = bb_dim_arr
        param_data['tf_cam_ego'] = np.reshape(self.tf_cam_ego, (16,))
        self.logger.write_params(param_data)
        ###################################

        print("Post-processing data now")
        # loop over all the actually seen ado objects, and sort them by class. this way we know the max number of candidates for coorespondences
        for ado_name in self.ado_names:
            class_str = self.ado_name_to_class[ado_name]
            self.class_str_to_name_dict[class_str].append(ado_name)

        t_img_to_t_est_dict = {}
        add_errs = []
        R_errs = []
        t_errs = []
        tms = []
        for i, t_est in enumerate(self.t_est):
            if t_est < 0:
                continue

            # find corresponding ego pose 
            t_gt, gt_ind = find_closest_by_time(t_est, self.ego_gt_time_pose)
            tf_w_ego_gt = pose_to_tf(self.ego_gt_pose[gt_ind])
            t_est2, est_ind = find_closest_by_time(t_est, self.ego_est_time_pose)
            tf_w_ego_est = pose_to_tf(self.ego_est_pose[est_ind])
            # print("t_gt - t_est = {.3f} s, t_est2 - t_est = {.3f} s".format(t_gt - t_est, t_est2 - t_est))
            tf_w_cam = tf_w_ego_gt @ inv_tf(self.tf_cam_ego)
            tf_cam_w = inv_tf(tf_w_cam)
            # pdb.set_trace()


            corespondences = []
            for class_name_seen in self.ado_est_pose_BY_TIME_BY_CLASS[t_est].keys():
                ado_name_candidates = self.class_str_to_name_dict[class_name_seen]
                total_num_of_this_classs = len(ado_name_candidates)
                num_seen_of_this_class = len(self.ado_est_pose_BY_TIME_BY_CLASS[t_est][class_name_seen])

                cost_mat = 1e5*np.ones((num_seen_of_this_class, total_num_of_this_classs))

                # get each tf_w_ado_est that we have this round (can be less than total number we have)
                ado_est_data_list = []
                ado_gt_data_list = []
                for hun_row, (tf_w_ado_est_ros_format, bb_proj, connected_inds, bb_proj_gt) in enumerate(self.ado_est_pose_BY_TIME_BY_CLASS[t_est][class_name_seen]): # ado_pose, bb_proj, connected_inds
                    tf_w_ado_est = pose_to_tf(tf_w_ado_est_ros_format)
                    ado_est_data_list.append((tf_w_ado_est, bb_proj, connected_inds, bb_proj_gt))

                    # get tf_w_ado_gt for each candidate
                    for hun_col, ado_name_cand in enumerate(ado_name_candidates):
                        t_gt, gt_ind = find_closest_by_time(t_est, self.t_gt[ado_name_cand])
                        tf_w_ado_gt = pose_to_tf(self.ado_gt_pose[ado_name_cand][gt_ind])
                        ado_gt_data_list.append((tf_w_ado_gt, t_gt, ado_name_cand))
                        cost_mat[hun_row, hun_col] = la.norm(tf_w_ado_gt[0:3, 3] - tf_w_ado_est[0:3, 3])
                        try:
                            assert(abs(t_gt - t_est) < 0.1) # make sure there are no surprises
                        except:
                            print("FAILED ASSERTION: assert(abs(t_gt - t_est) < 0.1) ...  abs(t_gt - t_est) = {}".format(abs(t_gt - t_est)))
                            pdb.set_trace()
                            raise RuntimeError("FAILED ASSERTION!!!")
                # now we have a cost matrix with the rows being the ado objects we have seen this round (but only know the classes of) and the columns being the ground truth ado ojbects (we know the full names in ) ado_name_candidates list
                row_inds, col_inds = scipy_hung_alg(cost_mat)

                # use our results to build tuples
                for (ado_seen_idx, ado_gt_idx) in zip(row_inds, col_inds):
                    tf_w_ado_est, bb_proj, connected_inds, bb_proj_gt = ado_est_data_list[ado_seen_idx]
                    tf_w_ado_gt, t_gt, ado_name = ado_gt_data_list[ado_gt_idx]

                    corespondences.append((tf_w_ado_est, tf_w_ado_gt, ado_name, class_name_seen, t_gt, bb_proj, connected_inds, bb_proj_gt))
                    
                    # print('error (trans dist) for {} after hung alg = {}'.format(ado_name, cost_mat[ado_seen_idx, ado_gt_idx]))
            ################ END HUNG ALG ##############################
            
            if len(corespondences) == 0:
                continue
            R_deltaz = np.array([[ np.cos(np.pi),-np.sin(np.pi), 0.              ],
                                [ np.sin(np.pi), np.cos(np.pi), 0.              ],
                                [ 0.             , 0.             , 1.              ]])



            for tf_w_ado_est, tf_w_ado_gt, name, class_str, t_gt, bb_proj, connected_inds, bb_proj_gt in corespondences:
                # if self.rb_name == "msl_raptor_output_from_bag_rosbag_for_post_process_2019-12-18-02-10-28.bag" and t_gt > 31:
                #     continue

                # if self.rb_name == "msl_raptor_output_from_bag_scene_2.bag" and t_gt > 2.3 and t_gt < 18:
                #     """ FIX ERROR IN GROUNT TRUTH!!!! """
                #     continue
                #     tf_w_ado_gt[0:3, 0:3] = R_deltaz @ tf_w_ado_gt[0:3, 0:3]
                # if self.rb_name == "msl_raptor_output_from_bag_scene_4.bag" and t_gt > 4.35 and t_gt < 7.7:
                #     """ FIX ERROR IN GROUNT TRUTH!!!! """
                #     continue


                log_data = {}
                box_length, box_width, box_height, diam = self.bb_3d_dict_all[name]
                vertices = np.array([[ box_length/2, box_width/2, box_height/2, 1.],
                                     [ box_length/2, box_width/2,-box_height/2, 1.],
                                     [ box_length/2,-box_width/2,-box_height/2, 1.],
                                     [ box_length/2,-box_width/2, box_height/2, 1.],
                                     [-box_length/2,-box_width/2, box_height/2, 1.],
                                     [-box_length/2,-box_width/2,-box_height/2, 1.],
                                     [-box_length/2, box_width/2,-box_height/2, 1.],
                                     [-box_length/2, box_width/2, box_height/2, 1.]]).T
                
                # if gt_ind > len(self.ego_gt_pose):
                #     break # this can happen at the end of a bag
                # pose_msg, _ = find_closest_by_time(t_est, self.ego_est_time_pose, message_list=self.ego_est_pose)
                # tf_w_ego_est = pose_to_tf(pose_msg)

                
                tf_cam_ado_est = tf_cam_w @ tf_w_ado_est
                tf_cam_ado_gt = tf_cam_w @ tf_w_ado_gt

                R_cam_ado_pr = tf_cam_ado_est[0:3, 0:3]
                t_cam_ado_pr = tf_cam_ado_est[0:3, 3].reshape((3, 1))
                tf_cam_ado_gt = tf_cam_w @ tf_w_ado_gt
                R_cam_ado_gt = tf_cam_ado_gt[0:3, 0:3]
                t_cam_ado_gt = tf_cam_ado_gt[0:3, 3].reshape((3, 1))
                
                ######################################################
                
                self.raptor_metrics.update_all_metrics(name=name, vertices=vertices, tf_w_cam=tf_w_cam, R_cam_ado_gt=R_cam_ado_gt, t_cam_ado_gt=t_cam_ado_gt, R_cam_ado_pr=R_cam_ado_pr, t_cam_ado_pr=t_cam_ado_pr, K=self.new_camera_matrix)

                # Write data to log file #############################
                log_data['time'] = t_est
                log_data['state_est'] = tf_to_state_vec(tf_w_ado_est)
                log_data['state_gt'] = tf_to_state_vec(tf_w_ado_gt)
                log_data['ego_state_est'] = tf_to_state_vec(tf_w_ego_est)
                log_data['ego_state_gt'] = tf_to_state_vec(tf_w_ego_gt)
                corners3D_pr = (tf_w_ado_est @ vertices)[0:3,:]
                corners3D_gt = (tf_w_ado_gt @ vertices)[0:3,:]
                log_data['corners_3d_est'] = np.reshape(corners3D_pr, (corners3D_pr.size,))
                log_data['corners_3d_gt'] = np.reshape(corners3D_gt, (corners3D_gt.size,))
                log_data['proj_corners_est'] = np.reshape(self.raptor_metrics.proj_2d_pr[name].T, (self.raptor_metrics.proj_2d_pr[name].size,))
                log_data['proj_corners_gt'] = np.reshape(self.raptor_metrics.proj_2d_gt[name].T, (self.raptor_metrics.proj_2d_gt[name].size,))

                log_data['x_err'] = tf_w_ado_est[0, 3] - tf_w_ado_gt[0, 3]
                log_data['y_err'] = tf_w_ado_est[1, 3] - tf_w_ado_gt[1, 3]
                log_data['z_err'] = tf_w_ado_est[2, 3] - tf_w_ado_gt[2, 3]
                log_data['ang_err'] = calcAngularDistance(tf_w_ado_est[0:3, 0:3], tf_w_ado_gt[0:3, 0:3])
                log_data['pix_err'] = np.mean(la.norm(self.raptor_metrics.proj_2d_pr[name] - self.raptor_metrics.proj_2d_gt[name], axis=0))
                log_data['add_err'] = np.mean(la.norm(corners3D_pr - corners3D_gt, axis=0))
                log_data['measurement_dist'] = la.norm(tf_w_ego_gt[0:3, 3] - tf_w_ado_gt[0:3, 3])
                add_errs.append(log_data['add_err'])
                R_errs.append(log_data['ang_err'])
                t_errs.append(la.norm(tf_w_ado_est[0:3, 3] - tf_w_ado_gt[0:3, 3]))
                tms.append(t_gt)

                if len(self.abb_time_list[name]) > 0:
                    (abb, im_seg_mode), _ = find_closest_by_time(t_est, self.abb_time_list[name], message_list=self.abb_list[name])
                    log_data['abb'] = abb
                    log_data['im_seg_mode'] = im_seg_mode
                self.logger.write_data_to_log(log_data, name, mode='est')
                self.logger.write_data_to_log(log_data, name, mode='gt')
                self.logger.write_data_to_log(log_data, name, mode='err')
                ######################################################
                # draw on image (3d bb estimate)
                if self.b_save_3dbb_imgs:
                    if t_est in self.processed_image_dict:
                        image, _, __ = self.processed_image_dict[t_est]
                    else:
                        img_msg, img_pos = find_closest_by_time(t_est, self.img_time_buffer, message_list=self.img_msg_buffer)
                        img_time = self.img_time_buffer[img_pos]
                        t_img_to_t_est_dict[img_time] = t_est
                        image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
                        image = cv2.undistort(image, self.K, self.dist_coefs, None, self.new_camera_matrix)
                    
                    
                    if self.b_save_3dbb_imgs and len(bb_proj) > 0:
                        image_to_draw_on = image
                        ###### COLOR LEGEND ##################################
                        # black/white is gt, "darker" colors are calculated locally
                        # "light" color - this is msl-raptor's estimate as calculated in real time (in msl-raptor code)
                        # "darker" color - this is the estimate calculated locally
                        # black - this is the gt calculated locally
                        # white - this is the gt calculated in msl raptor
                        color_est_raptor = self.ado_name_to_color[name]
                        color_est_local  = (self.ado_name_to_color[name][0] // 2, self.ado_name_to_color[name][1] // 2, self.ado_name_to_color[name][2] // 2)
                        color_gt_local   = (0, 0, 0)  # black
                        color_gt_raptor  = (255, 255, 2550)  # white 
                        b_draw_est_raptor = False
                        b_draw_est_local  = False
                        b_draw_gt_raptor  = False
                        b_draw_gt_local   = True

                        # draw the gt verts if this is enabled
                        if self.b_plot_gt_overlay:
                            if b_draw_gt_raptor and len(bb_proj_gt) > 0:
                                # if sent over, plot the gt projection as calculated by msl raptor
                                image_to_draw_on = draw_2d_proj_of_3D_bounding_box(image_to_draw_on, bb_proj_gt, color_pr=color_gt_raptor, linewidth=self.bb_linewidth, b_verts_only=False, inds_to_connect=connected_inds)

                            # plot the verts as calculated here
                            bb_3d, _, _, _, _, _, connected_inds_gt_list = self.info_for_gt_overlay # bb_3d, obj_width, obj_height, classes_names, classes_ids, objects_names_per_class, connected_inds
                            if not class_str in bb_3d:
                                print("WARNING - HAVENT TESTED THIS YET AND I THINK IT IS OUTDATED")
                                bb_proj_gt_calc_local = np.fliplr(pose_to_3d_bb_proj(tf_w_ado_gt, tf_w_ego_gt, vertices, self.camera)) # fliplr is needed because of x /y  <===> column / row
                                pdb.set_trace()
                            else:
                                bb_proj_gt_calc_local = np.fliplr(pose_to_3d_bb_proj(tf_w_ado_gt, tf_w_ego_gt, bb_3d[class_str], self.camera) ) # fliplr is needed because of x /y  <===> column / row
                                bb_proj_est_calc_local = np.fliplr(pose_to_3d_bb_proj(tf_w_ado_est, tf_w_ego_est, bb_3d[class_str], self.camera) )
                            if b_draw_gt_local:
                                image_to_draw_on = draw_2d_proj_of_3D_bounding_box(image_to_draw_on, bb_proj_gt_calc_local, color_pr=color_gt_local, linewidth=self.bb_linewidth, b_verts_only=False, inds_to_connect=connected_inds)
                            if b_draw_est_local:
                                image_to_draw_on = draw_2d_proj_of_3D_bounding_box(image_to_draw_on, bb_proj_est_calc_local, color_pr=color_est_local, linewidth=self.bb_linewidth, b_verts_only=False, inds_to_connect=connected_inds)


                        # now draw our estimated verts - as calculated by msl raptor
                        if t_est in self.processed_image_dict:
                            if b_draw_est_raptor:
                                image_to_draw_on = draw_2d_proj_of_3D_bounding_box(image_to_draw_on, bb_proj, color_pr=color_est_raptor, linewidth=self.bb_linewidth, b_verts_only=False, inds_to_connect=connected_inds)
                            self.processed_image_dict[t_est][0] = image_to_draw_on
                            self.processed_image_dict[t_est][1].append(bb_proj)
                            self.processed_image_dict[t_est][2].append(name)
                        else:
                            if b_draw_est_raptor:
                                image_to_draw_on = draw_2d_proj_of_3D_bounding_box(image_to_draw_on, bb_proj, color_pr=color_est_raptor, linewidth=self.bb_linewidth, b_verts_only=False, inds_to_connect=connected_inds)
                            self.processed_image_dict[t_est] = [image_to_draw_on, [bb_proj], [name]]


                        # if name=="swell_bottle":
                        #     if i == 0:
                                # pdb.set_trace()
                                # print("tf_w_ado_gt:\n{}".format(tf_w_ado_gt))
                            # print("tf_w_ego_gt:\n{}".format(tf_w_ego_gt))
                            # print("tf_w_ego_est:\n{}".format(tf_w_ego_est))
                            # t_err_ego = la.norm(tf_w_ego_gt[0:3, 3] - tf_w_ego_est[0:3, 3])
                            # R_err_ego = calcAngularDistance(tf_w_ego_gt[0:3, 0:3], tf_w_ego_est[0:3, 0:3]) # in degrees
                            # # print("ego err: trans = {:.2f} mm, rot = {:.3f} deg".format(t_err_ego*1000, R_err_ego))
                            # print("tf_w_ado_est:\n{}".format(tf_w_ado_est))
                            # print("tf_w_ado_gt:\n{}".format(tf_w_ado_gt))
                            # pdb.set_trace()
                            # if i == 3:
                            #     pdb.set_trace()
            # save the image
            fn_str = "mslraptor_{:d}".format(i)
            cv2.imwrite("/mounted_folder/raptor_processed_bags/output_imgs/" + fn_str + ".jpg", image_to_draw_on)
            # pdb.set_trace()

                    ######################################################
            

        if self.raptor_metrics is not None:
            self.raptor_metrics.calc_final_metrics()
            self.raptor_metrics.print_final_metrics()

        # # write images!!
        # if self.b_save_3dbb_imgs:
        #     b_fill_in_gaps = False
        #     img_ind = 0
        #     max_skip_count = 3
        #     if b_fill_in_gaps:
        #         for img_msg_time, img_msg in zip(self.img_time_buffer, self.img_msg_buffer):
        #             if img_msg_time < -0.04:
        #                 continue
        #             if self.rb_name in self.bags_and_cut_times and img_msg_time + self.t0 > self.bags_and_cut_times[self.rb_name]:
        #                 break
        #             if img_msg_time in t_img_to_t_est_dict:
        #                 # we have a bb for this frame
        #                 image, last_bb_proj_list, last_name_list = self.processed_image_dict[t_img_to_t_est_dict[img_msg_time]]
        #                 skip_count = 0
        #             else:
        #                 # we dont, use the latest bb if within skip_count
        #                 image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        #                 image = cv2.undistort(image, self.K, self.dist_coefs, None, self.new_camera_matrix)
        #                 if skip_count < max_skip_count: # if its jus a little later... reuse prev box
        #                     for bb_proj, name in zip(last_bb_proj_list, last_name_list):
        #                         image = draw_2d_proj_of_3D_bounding_box(image, bb_proj, color_pr=self.ado_name_to_color[name], linewidth=self.bb_linewidth)
        #                 else:
        #                     print("over skip count ({})".format(img_ind))
        #                 skip_count += 1
        #             fn_str = "mslraptor_{:d}".format(img_ind)
        #             cv2.imwrite("/mounted_folder/raptor_processed_bags/output_imgs/" + fn_str + ".jpg", image)
        #             img_ind += 1
        #     else:
        #         for img_ind, t_est in enumerate(self.processed_image_dict):
        #             image, _, _ = self.processed_image_dict[t_est]
        #             fn_str = "mslraptor_{:d}".format(img_ind)
        #             cv2.imwrite("/mounted_folder/raptor_processed_bags/output_imgs/" + fn_str + ".jpg", image)
        
        print("done processing rosbag into logs!")
        plt.figure(0)
        plt.plot(tms, add_errs, 'b.')
        plt.gca().set_title("ADD")
        plt.figure(1)
        plt.plot(tms, t_errs, 'r.')
        plt.gca().set_title("Trans Err")
        plt.figure(2)
        plt.plot(tms, R_errs, 'm.')
        plt.gca().set_title("Rotation Err")
        plt.show(block=False)
        input("\nPress enter to close program\n")


    def process_rb(self):
        """
        Reads all data from the rosbag (since time ordering is not guaranteed). Processes each message based on topic
        """
        print("Processing {}".format(self.rb_name))
        for i, (topic, msg, t) in enumerate(self.bag.read_messages()):
            t_split = topic.split("/")
            if topic in self.topic_func_dict:
                self.topic_func_dict[topic](msg, t=t.to_sec())
            elif t_split[-1] == 'msl_raptor_state': # estimate
                self.parse_ado_est_msg(msg)
            elif t_split[1] in self.ado_names_all and t_split[-1] == 'pose' and t_split[-2] == 'vision_pose': # ground truth from a quad (mavros) / nocs
                name = t_split[1]
                if name == self.ego_quad_ns.split('/')[-1] or name == "quad7quad7":
                    continue
                self.ado_names.add(name)
                self.parse_ado_gt_msg(msg, name=name, t=t.to_sec())
            elif (t_split[1] == 'vrpn_client_node' and t_split[-1] == 'pose'): # ground truth from optitrack default 
                name = t_split[2]
                if name == self.ego_quad_ns.split('/')[-1] or name == "quad7quad7":
                    continue
                self.ado_names.add(name)
                self.parse_ado_gt_msg(msg, name=name, t=t.to_sec())

        self.t_est = np.sort(list(self.t_est))
        self.t0 = np.min(self.t_est)
        self.tf = np.max(self.t_est)
        self.t_est -= self.t0
        old_keys = copy(list(self.ado_est_pose_BY_TIME_BY_CLASS.keys()))
        for old_key in old_keys:
            self.ado_est_pose_BY_TIME_BY_CLASS[old_key - self.t0] = self.ado_est_pose_BY_TIME_BY_CLASS.pop(old_key)
        for i, n in enumerate(self.ado_names):
            self.t_gt[n] = np.asarray(self.t_gt[n]) - self.t0
            self.ado_name_to_color[n] = self.color_list[i]
        self.img_time_buffer = np.asarray(self.img_time_buffer) - self.t0
        # self.detect_time[n] = np.asarray(self.detect_time) - self.t0
        # self.detect_times[n] = np.asarray(self.detect_times) - self.t0
        self.ego_est_time_pose = np.asarray(self.ego_est_time_pose) - self.t0
        self.ego_gt_time_pose = np.asarray(self.ego_gt_time_pose) - self.t0


    def read_yaml(self, ego_yaml="quad7", ado_yaml="all_obs"):
        yaml_path="/root/msl_raptor_ws/src/msl_raptor/params/"
        with open(yaml_path + ego_yaml + '.yaml', 'r') as stream:
            try:
                # this means its the ego robot, dont add it to ado (but get its camera params)
                obj_prms = yaml.safe_load(stream)
                self.ego_name = obj_prms['ns']
                self.tf_cam_ego = np.eye(4)
                self.tf_cam_ego[0:3, 3] = np.asarray(obj_prms['t_cam_ego'])
                self.tf_cam_ego[0:3, 0:3] = np.reshape(obj_prms['R_cam_ego'], (3, 3))
                Angle_x = obj_prms['dx']
                Angle_y = obj_prms['dy']
                Angle_z = obj_prms['dz']
                R_deltax = np.array([[ 1.             , 0.             , 0.              ],
                                     [ 0.             , np.cos(Angle_x),-np.sin(Angle_x) ],
                                     [ 0.             , np.sin(Angle_x), np.cos(Angle_x) ]])
                R_deltay = np.array([[ np.cos(Angle_y), 0.             , np.sin(Angle_y) ],
                                     [ 0.             , 1.             , 0               ],
                                     [-np.sin(Angle_y), 0.             , np.cos(Angle_y) ]])
                R_deltaz = np.array([[ np.cos(Angle_z),-np.sin(Angle_z), 0.              ],
                                     [ np.sin(Angle_z), np.cos(Angle_z), 0.              ],
                                     [ 0.             , 0.             , 1.              ]])
                R_delta = R_deltax @ R_deltay @ R_deltaz
                self.tf_cam_ego[0:3, 0:3] = np.matmul(R_delta, self.tf_cam_ego[0:3, 0:3])
            except yaml.YAMLError as exc:
                print(exc)


        with open(yaml_path + ado_yaml + '.yaml', 'r') as stream:
            try:
                obj_prms = list(yaml.load_all(stream))
                for obj_dict in obj_prms:
                    name = obj_dict['ns']
                    self.b_enforce_0_dict[name] = defaultdict(bool)  # false by default
                    self.fixed_vals[name] = defaultdict(dict)
                    
                    class_str = obj_dict['class_str']
                    self.ado_names_all.append(name)
                    l = float(obj_dict['bound_box_l']) 
                    w = float(obj_dict['bound_box_w'])
                    h = float(obj_dict['bound_box_h'])
                    # self.class_str_to_name_dict[class_str].append(name)
                    self.ado_name_to_class[name] = class_str
                    if 'diam' in obj_dict:
                        d = obj_dict['diam']
                    else:
                        d = la.norm([l, w, h])
                    self.bb_3d_dict_all[name] = np.array([l, w, h, d])
                    for el in obj_dict['b_enforce_0']:
                        self.b_enforce_0_dict[name][el] = True
                        additional_key = 'fixed_' + el
                        if additional_key in obj_dict:
                            self.fixed_vals[name][el] = obj_dict[additional_key]

            except yaml.YAMLError as exc:
                print(exc)


    def parse_camera_img_msg(self, msg, t:None):
        if t is None:
            t_img = msg.header.stamp.to_sec()
        else:
            t_img = t
        self.img_time_buffer.append(t_img)
        self.img_msg_buffer.append(msg)


    def parse_camera_info_msg(self, msg, t=None):
        if self.K is None:
            camera_info = msg
            self.K = np.reshape(camera_info.K, (3, 3))
            if len(camera_info.D) == 5:  # this just checks if there are distortion coefficients included in the message
                self.dist_coefs = np.reshape(camera_info.D, (5,))
                self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist_coefs, (camera_info.width, camera_info.height), 0, (camera_info.width, camera_info.height))
            else:
                self.dist_coefs = None
                self.new_camera_matrix = self.K
            if self.b_plot_gt_overlay:
                self.camera = camera_slim(camera_info, self.tf_cam_ego, self.K, self.new_camera_matrix)
    

    def parse_ado_est_msg(self, msg, t=None):
        """
        record estimated poses from msl-raptor
        """
        tracked_obs = msg.tracked_objects
        if len(tracked_obs) == 0:
            return

        t_est = None
        if t is not None:
            t_est = t

        for to in tracked_obs:
            if t_est is None:
                t_est = to.pose.header.stamp.to_sec() # to message doesnt have its own header, the publisher set this time to be that of the image the measurement that went into the ukf was received at
            pose = to.pose.pose

            connected_inds = []
            if len(to.connected_inds) > 0:
                connected_inds = np.reshape(to.connected_inds, (int(len(to.connected_inds)/2), 2) )
            else:
                # this means we didnt have custom vertices, use generic 3D bounding box
                connected_inds = np.array([[0, 1], [1, 2], [2, 3], [3, 0],  # edges of front surface of 3D bb (starting at "upper left" and going counter-clockwise while facing the way the object is)
                                           [7, 4], [4, 5], [5, 6], [6, 7],  # edges of back surface of 3D bb (starting at "upper left" and going counter-clockwise while facing the way the object is)
                                           [0, 7], [1, 6], [2, 5], [3, 4]]) # horizontal edges of 3D bb (starting at "upper left" and going counter-clockwise while facing the way the object is)

            proj_3d_bb = []
            proj_3d_bb_gt = []
            if len(to.projected_3d_bb) > 0:
                proj_3d_bb = np.reshape(to.projected_3d_bb, (int(len(to.projected_3d_bb)/2), 2) )
                
                if np.max(connected_inds)*1.5 < proj_3d_bb.shape[0]:
                    # this means we have "too many" vertices, and indicates we lumped both estimate and gt together (via np.vstack((est, gt)))
                    # this should be true cause connected_ind's contains vertex indices, so its max value should be the number of points
                    proj_3d_bb_gt = proj_3d_bb[proj_3d_bb.shape[0]//2:, :]   # first half is the estimates
                    proj_3d_bb    = proj_3d_bb[0:proj_3d_bb.shape[0]//2, :]  # second half is the ground truth


            if to.class_str in self.ado_est_pose_BY_TIME_BY_CLASS[t_est]:
                self.ado_est_pose_BY_TIME_BY_CLASS[t_est][to.class_str].append((pose, proj_3d_bb, connected_inds, proj_3d_bb_gt))
            else:
                self.ado_est_pose_BY_TIME_BY_CLASS[t_est][to.class_str] = [(pose, proj_3d_bb, connected_inds, proj_3d_bb_gt)]

        self.t_est.add(t_est)


    def parse_ado_gt_msg(self, msg, name, t=None):
        """
        record optitrack poses of tracked quad
        """
        if np.isnan(msg.pose.orientation.x):
            return  # if the message is invalid, ignore it
        if t is None:
            self.t_gt[name].append(msg.header.stamp.to_sec())
        else:
            self.t_gt[name].append(t)
        
        pose = enforce_constraints_pose(msg.pose, self.b_enforce_0_dict[name], self.fixed_vals[name])
        self.ado_gt_pose[name].append(pose)

        
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
        for msg in msg.boxes:
            name = msg.class_str
            t = msg.header.stamp.to_sec()
            self.abb_list[name].append(([msg.x, msg.y, msg.width, msg.height, msg.angle*180./np.pi], msg.im_seg_mode))
            self.abb_time_list[name].append(t)

class camera_slim: # THIS IS A SLIMMED DOWN VERSION OF THE CLASS FROM RAPTOR. We get the inputs from our yaml file (this is for debugging and allows us to plot 3d verts)
    def __init__(self, camera_info, tf_cam_ego, K, new_camera_matrix):
        """
        K: camera intrinsic matrix 
        tf_cam_ego: camera pose relative to the ego_quad (fixed)
        """
        self.K = K
        if len(camera_info.D) == 5:
            self.dist_coefs = np.reshape(camera_info.D, (5,))
            self.new_camera_matrix = new_camera_matrix
        else:
            self.dist_coefs = None
            self.new_camera_matrix = new_camera_matrix

        self.K_inv = la.inv(self.K)
        self.new_camera_matrix_inv = la.inv(self.new_camera_matrix)
        self.tf_cam_ego = tf_cam_ego

    def pnt3d_to_pix(self, pnt_c):
        """
        input: assumes pnt in camera frame
        output: [row, col] i.e. the projection of xyz onto camera plane
        """
        rc = self.new_camera_matrix @ np.reshape(pnt_c[0:3], 3, 1)
        rc = np.array([rc[1], rc[0]]) / rc[2]
        return rc


if __name__ == '__main__':
    try:
        if len(sys.argv) == 6:
            my_rb_name = sys.argv[1]
            my_data_source = sys.argv[2]
            my_ego_yaml = sys.argv[3]
            my_ado_yaml = sys.argv[4]
            my_b_save_3dbb_imgs = bool(sys.argv[5])
        else:
            raise RuntimeError("Incorrect arguments! needs <rosbag_name> <data_source> <ego_yaml> <ado_yaml> <b_save_3dbb_imgs> (leave off .bag and .yaml extensions)")
        np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
        program = rosbags_to_logs(rb_name=my_rb_name, data_source=my_data_source, ego_yaml=my_ego_yaml, ado_yaml=my_ado_yaml, b_save_3dbb_imgs=my_b_save_3dbb_imgs)
        
    except:
        import traceback
        traceback.print_exc()
