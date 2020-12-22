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
    def __init__(self, rb_path_and_name, ego_pose_topic, cal_board_topic, matlab_data, camera_image_topic="/camera/image_raw", b_write_images_from_rosbag=False):
        bag_in = rosbag.Bag(rb_path_and_name + ".bag", 'r')
        cv_bridge = CvBridge()
        image_time_dict = {}
        ego_pose_time_dict = {}
        cal_board_time_dict = {}
        all_data = {}
        for topic, msg, t in bag_in.read_messages():
            time = t.to_sec()
            if topic == camera_image_topic:
                image_time_dict[time] = msg
            elif ego_pose_topic and topic == ego_pose_topic:
                ego_pose_time_dict[time] = msg
            elif cal_board_topic and topic == cal_board_topic:
                cal_board_time_dict[time] = msg
        
        print("done reading in bag, now syncing times & calculating extrinics")
        

        # Manually calculate fixed offset between optitrack board frame (bo) and matlab's assumed board frame (b)
        perp_dist_board_to_center_optitrack_ball = 0.009  # its a little less than 1 cm
        t_b_bo = np.asarray([  7 * 40.0 / 1000.0, 
                             4.5 * 40.0 / 1000.0, 
                             -perp_dist_board_to_center_optitrack_ball])  # pointing to bo from b, expressed in b frame
        R_b_bo = np.array([[ 0.,  1.,  0.],
                           [-1.,  0.,  0.],
                           [ 0.,  0.,  1.]])  # matlabs x is our y, matlabs y is our z, matlabs z is our x
        tf_b_bo_CONST = t_and_R_to_tf(t_b_bo, R_b_bo)
        ######################################################################################

        # Init loop variables
        chosen_pic_ind = 0
        tf_e_c_ave = np.zeros((3,))
        num_cal_imgs = matlab_data['rot_vec_c_b_arr'].shape[0]
        quat_e_c = np.zeros((num_cal_imgs,4))
        tf_w_bo_CONST = None # this value should be constant, but optitrack flips it due to symetry of the marker placement. Only use first value to be consistant
        outputs = []
        for img_ind, img_time in enumerate(image_time_dict):
            if not b_write_images_from_rosbag and not img_ind in matlab_data['used_img_inds']:
                continue
            image = cv_bridge.imgmsg_to_cv2(image_time_dict[img_time], desired_encoding="passthrough")
            closest_ego_time, _ = find_closest_by_time(img_time, [*ego_pose_time_dict.keys()]) 
            closest_cal_board_time, _ = find_closest_by_time(img_time, [*cal_board_time_dict.keys()])  
            all_data[(img_time + closest_ego_time + closest_cal_board_time) / 3.0] = (image, ego_pose_time_dict[closest_ego_time], cal_board_time_dict[closest_cal_board_time])
            if b_write_images_from_rosbag:
                fn_str = "cal_img_{:06}".format(img_ind)
                cv2.imwrite(('/').join(rb_path_and_name.split('/')[:-1]) + '/' + fn_str + ".jpg", image)
            
            if img_ind in matlab_data['used_img_inds']:
                print("\n###############   ind {}  ({}/{})  ################".format(img_ind, chosen_pic_ind, num_cal_imgs))
                # Extract ego (and cal board) pose from optitrack data (matched by time to the image with correct index)
                t_w_e = np.array([ego_pose_time_dict[closest_ego_time].pose.position.x, ego_pose_time_dict[closest_ego_time].pose.position.y, ego_pose_time_dict[closest_ego_time].pose.position.z])
                R_w_e = quat_to_rotm(np.array([ego_pose_time_dict[closest_ego_time].pose.orientation.w,
                                               ego_pose_time_dict[closest_ego_time].pose.orientation.x,
                                               ego_pose_time_dict[closest_ego_time].pose.orientation.y,
                                               ego_pose_time_dict[closest_ego_time].pose.orientation.z]))
                tf_w_e = t_and_R_to_tf(t_w_e, R_w_e)
                if tf_w_bo_CONST is None:
                    t_w_bo = np.array([cal_board_time_dict[closest_cal_board_time].pose.position.x, cal_board_time_dict[closest_cal_board_time].pose.position.y, cal_board_time_dict[closest_cal_board_time].pose.position.z])
                    R_w_bo = quat_to_rotm(np.array([cal_board_time_dict[closest_cal_board_time].pose.orientation.w,
                                                    cal_board_time_dict[closest_cal_board_time].pose.orientation.x,
                                                    cal_board_time_dict[closest_cal_board_time].pose.orientation.y,
                                                    cal_board_time_dict[closest_cal_board_time].pose.orientation.z]))
                    tf_w_bo_CONST = t_and_R_to_tf(t_w_bo, R_w_bo)
                    print("tf_w_bo_CONST:\n{}".format(tf_w_bo_CONST))
                    print("tf_b_bo_CONST:\n{}".format(tf_b_bo_CONST))
                
                # Calculate board pose in ego frame
                tf_e_bo = inv_tf(tf_w_e) @ tf_w_bo_CONST
                
                # Extract relative board/camera pose from matlab data
                t_c_b = matlab_data['t_c_b_arr'][chosen_pic_ind, :]
                ax_ang_c_b = matlab_data['rot_vec_c_b_arr'][chosen_pic_ind, :]  # this is ax/ang format where |vec| = ang, vec/|vec| is axis
                R_c_b = quat_to_rotm(axang_to_quat(ax_ang_c_b)[0])[0].T
                tf_c_b = t_and_R_to_tf(t_c_b, R_c_b)
                
                # Apply manual offset to account for difference between optitrack board frame (bo) and matlab's assumed board frame (b)
                tf_c_bo = tf_c_b @ tf_b_bo_CONST
                
                # Calculate the estimate of the constant offset between ego and camera frames
                tf_e_c = tf_e_bo @ inv_tf(tf_c_bo)

                # Aggregate estimates for averaging
                tf_e_c_ave += tf_e_c[0:3, 3]
                quat_e_c[chosen_pic_ind, :] = rotm_to_quat(tf_e_c[0:3, 0:3])
                outputs.append(tf_e_c)

                # Print out debugging statements
                print("tf_w_e:\n{}".format(tf_w_e))
                print("tf_e_bo:\n{}".format(tf_e_bo))
                print("tf_c_b:\n{}".format(tf_c_b))
                print("tf_c_bo:\n{}".format(tf_c_bo))
                print("tf_e_c:\n{}".format(tf_e_c))
                # print("tf_c_e:\n{}".format(inv_tf(tf_e_c)))

                chosen_pic_ind += 1
        tf_e_c_ave /= num_cal_imgs
        quat_e_c_ave, _ = average_quaternions(Q=quat_e_c, w=None) # Q is a Nx4 numpy mat, Q[i, :] = [w,x,y,z], Adapted from https://github.com/christophhagen/averaging-quaternions/blob/master/averageQuaternions.py

        tf_e_c_final = np.eye(4)
        tf_e_c_final[0:3, 0:3] = quat_to_rotm(quat_e_c_ave)
        tf_e_c_final[0:3, 3] = tf_e_c_ave
        print("\n\nFINAL EXTRINSICS CALIBRATION RESULT (tf_e_c):\n{}".format(tf_e_c_final))

        t_e_c_CAD = [0.13854437, -0.01504337, -0.06380886]
        R_e_c_CAD = np.array([[ 0.,  0.,  1.],
                              [-1.,  0.,  0.],
                              [ 0., -1.,  0.]])
        tf_e_c_CAD = t_and_R_to_tf(t_e_c_CAD, R_e_c_CAD)
        print("\n(expected result)\n{}".format(tf_e_c_CAD))

        # print("############ ALL tf_e_c one after another ################")
        # for tf in outputs:
        #     print("{}".format(tf))
        # pdb.set_trace()

        print("done with calibration!")
        bag_in.close()

def t_and_R_to_tf(t, R):
    tf = np.eye(4)
    tf[0:3, 0:3] = R
    tf[0:3, 3] = t
    return tf


if __name__ == '__main__':
    try:
        np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
        # program = rosbags_to_logs(rb_name=my_rb_name, data_source=my_data_source, ego_yaml=my_ego_yaml, ado_yaml=my_ado_yaml, b_save_3dbb_imgs=my_b_save_3dbb_imgs)
        rb_path = "/mounted_folder/rosbags_for_post_process/camera_cal/"
        rb_path_and_name_ = rb_path + "rosbag_for_post_process_2020-12-03-12-43-17"
        camera_image_topic_ = "/quad7/camera/image_raw"
        ego_pose_topic_ = "/vrpn_client_node/quad7/pose"
        cal_board_topic_ = "/vrpn_client_node/checkerboard/pose"
        b_write_images_from_rosbag_ = False


        ########################### DATA FROM MATLAB (MANUALLY ENTERED) ###########################
        matlab_data_ = {}
        matlab_data_['used_img_inds'] = [0, 86, 97, 169, 239, 260, 378, 414, 693, 757, 830, 921, 964, 981, 1074]
        matlab_data_['t_c_b_arr'] = np.array([
                                    [-204.567770034014, -217.677474042156, 990.032277314905],
                                    [-241.546404051251, -126.740236866132, 709.886713521358],
                                    [-85.6310311412221, -129.233327306724, 686.263588428402],
                                    [-49.5758053954742, -132.259975949021, 936.297851125760],
                                    [-511.236399979537, -54.9621466716082, 824.336873995614],
                                    [-291.335427801684,  145.662553091104, 822.042529467022],
                                    [-282.889886937005, -295.942293810267, 931.881235124768],
                                    [ 232.219150237119, -389.795359703377, 893.802257101281],
                                    [-203.098056959464,  93.9557920606624, 935.056631288139],
                                    [-486.302207460492, -264.304177557648, 1343.08246820740],
                                    [-191.822344253332,  115.321836772336, 763.790317208289],
                                    [ 28.1456326667861, -34.1502267383312, 882.525520256375],
                                    [-371.202271462141, -19.6213934697185, 503.971991759718],
                                    [-278.611678789423,  50.0776274325494, 517.101170783092],
                                    [-191.235658748160, -335.979228257803, 845.450627083922]]) / 1000.0
        matlab_data_['rot_vec_c_b_arr'] = np.array([
                                    [0.00181597261003776, 0.0263438296488361,  0.00315630643979066],
                                    [0.219171868820300,   0.101821265610930,   0.00895061055771904],
                                    [0.217482956771523,   0.252397193996283,   0.0406677022725875],
                                    [0.433153576264349,   0.357197172355445,   0.186383416969125],
                                    [0.481991088967751,  -0.634258056173375,  -0.131816233343635],
                                    [0.200232296509818,  -0.519647497045448,  -0.174587275535448],
                                    [0.510429140862375,   0.321592318974736,  -0.258958029181621],
                                    [0.699048787090123,   0.550850565569477,   0.735999116910402],
                                    [0.0275119582924590,  0.263313674840271,   0.0212619309388616],
                                    [0.537074857096390,  -0.450167897583607,  -0.120763660764963],
                                    [0.378723765564576,   0.0806904468493387, -0.0410019979583945],
                                    [0.426746444624424,   0.792878667108163,  -0.0446109676724370],
                                    [0.538798448045761,  -0.278556496690573,  -0.0398432508792724],
                                    [0.419179991478450,  -0.566497313420050,  -0.184356571858263],
                                    [0.154521328121619,   0.0274436590158075, -0.0108059370684502]])
        ########################### END DATA FROM MATLAB ###########################

        extract_camera_cal_info_from_rosbag(rb_path_and_name=rb_path_and_name_, 
                                            ego_pose_topic=ego_pose_topic_, 
                                            cal_board_topic=cal_board_topic_, 
                                            matlab_data=matlab_data_, 
                                            camera_image_topic=camera_image_topic_, 
                                            b_write_images_from_rosbag=b_write_images_from_rosbag_)
    except:
        import traceback
        traceback.print_exc()
    print("done with program!")