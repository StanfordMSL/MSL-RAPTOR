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

class extract_camera_cal_info_from_rosbag:
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
            elif topic == "/quad7/camera/camera_info":
                pdb.set_trace()
        
        print("done reading in bag, now syncing times")
        pdb.set_trace()

        
        cv_bridge = CvBridge()
        all_data = {}
        for img_ind, img_time in enumerate(image_time_dict):
            image = cv_bridge.imgmsg_to_cv2(image_time_dict[img_time], desired_encoding="passthrough")
            # image = cv_bridge.imgmsg_to_cv2(image_time_dict[img_time], desired_encoding="bgr8")
            closest_ego_time, _ = find_closest_by_time(img_time, [*ego_pose_time_dict.keys()]) 
            closest_cal_board_time, _ = find_closest_by_time(img_time, [*cal_board_time_dict.keys()])  
            all_data[(img_time + closest_ego_time + closest_cal_board_time) / 3.0] = (image, ego_pose_time_dict[closest_ego_time], cal_board_time_dict[closest_cal_board_time])
            fn_str = "cal_img_{}".format(img_ind)
            cv2.imwrite(('/').join(rb_path_and_name.split('/')[:-1]) + '/' + fn_str + ".jpg", image)
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