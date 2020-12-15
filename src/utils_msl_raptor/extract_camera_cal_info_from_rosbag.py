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
from ros_utils import *
from math_utils import *

class extract_camera_cal_info_from_rosbag:
    '''
    FRAME NOTATION:
    t_A_B: translation vector from A to B in A frame
    R_A_B: rotation matrix from A to B in A frame
    tf_A_B: transfermation matrix s.t. a point in frame A (p_A) is p_A = tf_A_B * p_B
    w/W: world frame, defined by optitrack (x is pointing in long dir of flightroom, z is up)
    e/E: ego frame for quad where camera is, defined as center of optitrack marker on quads. This is located ~center of quad (left/right & front/back i.e. y_W & x_W dirs) and a few inches below the propellers
    c/C: camera frame
    bo/Bo: board frame (as in checkerboard) as defined by optitrack. origin is at average of the 4 corner optitrack markers, when looking perp. to board x is "up", y is "right", and z is "pointing away from you" (SOMETIMES FLIPS)
    b/B: board frame - matlab assumes coordinate frame for checker board is in upper left corner with positive x "to the right" and positive y "down" and positive z "into the board"
    '''
    def __init__(self, rb_path_and_name, camera_image_topic="/camera/image_raw", ego_pose_topic=None, cal_board_topic=None):
        bag_in = rosbag.Bag(rb_path_and_name + ".bag", 'r')
        image_time_dict = {}
        ego_pose_time_dict = {}
        cal_board_time_dict = {}
        for topic, msg, t in bag_in.read_messages():
            time = t.to_sec()
            if topic == camera_image_topic:
                image_time_dict[time] = msg
            elif ego_pose_topic and topic == ego_pose_topic:
                ego_pose_time_dict[time] = msg
            elif cal_board_topic and topic == cal_board_topic:
                cal_board_time_dict[time] = msg
        
        print("done reading in bag, now syncing times")
        
        cv_bridge = CvBridge()
        all_data = {}
        # MANUALLY ENTERED:
        used_imgs = [0, 86, 97, 169, 239, 260, 378, 414, 693, 757, 830, 921, 964, 981, 1074]
        t_c_b_arr = np.array([
                                [-204.57, -217.68,  990.03],
                                [-191.24, -335.98,  845.45],
                                [-49.576, -132.26,   936.3],
                                [-511.24, -54.962,  824.34],
                                [-291.34,  145.66,  822.04],
                                [-282.89, -295.94,  931.88],
                                [232.22,  -389.8,   893.8],
                                [-203.1,  93.956,  935.06],
                                [-486.3,  -264.3,  1343.1],
                                [-191.82,  115.32,  763.79],
                                [-241.55, -126.74,  709.89],
                                [28.146,  -34.15,  882.53],
                                [-371.2, -19.621,  503.97],
                                [-85.631, -129.23,  686.26],
                                [-278.61,  50.078,   517.1]]) / 1000.0
        rot_vec_c_b_arr = np.array([[0.001816,  0.026344, 0.0031563],
                                [0.15452,  0.027444, -0.010806],
                                [0.43315,  0.3572,  .18638],
                                [0.48199,  -0.63426,  -0.13182],
                                [0.20023,  -0.51965,  -0.17459],
                                [0.51043,  .32159,  -0.25896],
                                [0.69905,  .55085,   0.736],
                                [0.027512,  .26331,  0.021262],
                                [0.53707,  -0.45017,  -0.12076],
                                [0.37872,  .08069, -0.041002],
                                [0.21917,  .10182, 0.0089506],
                                [0.42675,  .79288, -0.044611],
                                [0.5388,  -0.27856, -0.039843],
                                [0.21748,  0.2524,  0.040668],
                                [0.41918,  0.5665,  -0.18436]])
        chosen_pic_ind = 0
        tf_e_c_ave = np.zeros((3,))
        N = rot_vec_c_b_arr.shape[0]
        quat_e_c = np.zeros((N,4))
        perp_dist_board_to_center_optitrack_ball = 0.009  # its a little less than 1 cm
        t_b_bo_offset = np.asarray([7*40.0/1000.0, 4.5*40.0/1000.0, -perp_dist_board_to_center_optitrack_ball])  # matlab assumes calibration board axis is in bottom right corner, each square is 40mm and board is 9 x 14
        # R_w_cal_offset = np.array([[ 0.,  0.,  1.],
        #                            [ 0., -1.,  0.],
        #                            [ 1.,  0.,  0.]])  # FOR SOME REASON THIS IS NEEDED???
        # R_cal_calmatlab = np.array([[ 0.,  0.,  1.],
        #                             [ 1.,  0.,  0.],
        #                             [ 0.,  1.,  0.]])  # matlabs x is our y, matlabs y is our z, matlabs z is our x
        R_b_bo = np.eye(3)
        R_b_bo = np.array([[ 0.,  1.,  0.],
                           [-1.,  0.,  0.],
                           [ 0.,  0.,  1.]])  # matlabs x is our y, matlabs y is our z, matlabs z is our x
        for img_ind, img_time in enumerate(image_time_dict):
            image = cv_bridge.imgmsg_to_cv2(image_time_dict[img_time], desired_encoding="passthrough")
            # image = cv_bridge.imgmsg_to_cv2(image_time_dict[img_time], desired_encoding="bgr8")
            closest_ego_time, _ = find_closest_by_time(img_time, [*ego_pose_time_dict.keys()]) 
            closest_cal_board_time, _ = find_closest_by_time(img_time, [*cal_board_time_dict.keys()])  
            all_data[(img_time + closest_ego_time + closest_cal_board_time) / 3.0] = (image, ego_pose_time_dict[closest_ego_time], cal_board_time_dict[closest_cal_board_time])
            fn_str = "cal_img_{}".format(img_ind)
            if 0:
                cv2.imwrite(('/').join(rb_path_and_name.split('/')[:-1]) + '/' + fn_str + ".jpg", image)
            
            if img_ind in used_imgs:
                t_w_e = np.array([ego_pose_time_dict[closest_ego_time].pose.position.x, ego_pose_time_dict[closest_ego_time].pose.position.y, ego_pose_time_dict[closest_ego_time].pose.position.z])
                t_w_bo = np.array([cal_board_time_dict[closest_cal_board_time].pose.position.x, cal_board_time_dict[closest_cal_board_time].pose.position.y, cal_board_time_dict[closest_cal_board_time].pose.position.z])
                R_w_e = quat_to_rotm(np.array([ego_pose_time_dict[closest_ego_time].pose.orientation.w,
                                                ego_pose_time_dict[closest_ego_time].pose.orientation.x,
                                                ego_pose_time_dict[closest_ego_time].pose.orientation.y,
                                                ego_pose_time_dict[closest_ego_time].pose.orientation.z]))
                R_w_bo = quat_to_rotm(np.array([cal_board_time_dict[closest_cal_board_time].pose.orientation.w,
                                                cal_board_time_dict[closest_cal_board_time].pose.orientation.x,
                                                cal_board_time_dict[closest_cal_board_time].pose.orientation.y,
                                                cal_board_time_dict[closest_cal_board_time].pose.orientation.z]))
                tf_w_e = np.eye(4)
                tf_w_e[0:3, 0:3] = R_w_e
                tf_w_e[0:3, 3] = t_w_e
                tf_w_bo = np.eye(4)
                tf_w_bo[0:3, 0:3] = R_w_bo
                tf_w_bo[0:3, 3] = t_w_bo
                tf_e_bo = inv_tf(tf_w_e) @ tf_w_bo
                
                t_c_b = t_c_b_arr[chosen_pic_ind, :]
                ax_ang_c_b = rot_vec_c_b_arr[chosen_pic_ind, :]  # this is ax/ang format where |vec| = ang, vec/|vec| is axis
                R_c_b = quat_to_rotm(axang_to_quat(ax_ang_c_b)[0])[0].T
                
                t_c_bo = t_c_b + t_b_bo_offset
                R_c_bo = R_c_b @ R_b_bo
                tf_c_bo = np.eye(4)
                tf_c_bo[0:3, 0:3] = R_c_bo
                tf_c_bo[0:3, 3] = t_c_bo
                
                tf_e_c = tf_e_bo @ inv_tf(tf_c_bo)

                tf_e_c_ave += tf_e_c[0:3, 3]
                quat_e_c[chosen_pic_ind, :] = rotm_to_quat(tf_e_c[0:3, 0:3])
                print("###############   ind {}  ({}/{})  ################".format(img_ind, chosen_pic_ind, N))
                print("tf_w_bo (CONSTANT):\n{}".format(tf_w_bo))
                print("tf_w_e:\n{}".format(tf_w_e))
                print("tf_e_bo:\n{}".format(tf_e_bo))
                print("tf_c_bo:\n{}".format(tf_c_bo))
                print("tf_e_c:\n{}".format(tf_e_c))
                if img_ind == 169:
                    pdb.set_trace()
                chosen_pic_ind += 1
        tf_e_c_ave /= N
        quat_e_c_ave, _ = average_quaternions(Q=quat_e_c, w=None)

        tf_e_c_final = np.eye(4)
        tf_e_c_final[0:3, 0:3] = quat_to_rotm(quat_e_c_ave)
        tf_e_c_final[0:3, 3] = tf_e_c_ave
        print("\n\nFINAL EXTRINSICS CALIBRATION RESULT:\n{}".format(tf_e_c_final))

        tf_e_c_CAD = np.eye(4)
        # tf_e_c_CAD[0:3, 0:3] = np.array([[ 0., -1.,  0.],
        #                                      [ 0.,  0., -1.],
        #                                      [ 1.,  0.,  0.]])
        tf_e_c_CAD[0:3, 0:3] = np.array([[ 0., -1.,  0.],
                                             [ 0.,  0., -1.],
                                             [ 1.,  0.,  0.]]).T
        tf_e_c_CAD[0:3, 3] = [0.13854437, -0.01504337, -0.06380886]
        print("\n(expected result)\n{}".format(tf_e_c_CAD))
        pdb.set_trace()
        # Adapted from https://github.com/christophhagen/averaging-quaternions/blob/master/averageQuaternions.py
        # Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
        # The quaternions are arranged as (w,x,y,z)

        print("done syncing data")
        bag_in.close()


if __name__ == '__main__':
    try:
        np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
        # program = rosbags_to_logs(rb_name=my_rb_name, data_source=my_data_source, ego_yaml=my_ego_yaml, ado_yaml=my_ado_yaml, b_save_3dbb_imgs=my_b_save_3dbb_imgs)
        rb_path = "/mounted_folder/rosbags_for_post_process/camera_cal/"
        rb_path_and_name_ = rb_path + "rosbag_for_post_process_2020-12-03-12-43-17"
        camera_image_topic_ = "/quad7/camera/image_raw"
        ego_pose_topic_ = "/vrpn_client_node/quad7/pose"
        cal_board_topic_ = "/vrpn_client_node/checkerboard/pose"

        extract_camera_cal_info_from_rosbag(rb_path_and_name=rb_path_and_name_, 
                                            camera_image_topic=camera_image_topic_, 
                                            ego_pose_topic=ego_pose_topic_, 
                                            cal_board_topic=cal_board_topic_)
    except:
        import traceback
        traceback.print_exc()
    print("done with program!")