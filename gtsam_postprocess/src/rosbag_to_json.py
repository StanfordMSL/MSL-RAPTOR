#!/usr/bin/env python3
# IMPORTS
# system
import sys, time, os
import pdb
# math
import numpy as np
# ros
# import rospy
import rosbag
from geometry_msgs.msg import PoseStamped, Twist, Pose
from msl_raptor.msg import AngledBbox,AngledBboxes,TrackedObjects,TrackedObject
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
# import tf
# libs & utils
import json
# sys.path.append(os.path.dirname("/root/msl_raptor_ws/src/msl_raptor/src/"))
# from utils_msl_raptor.ros_utils import *
# from utils_msl_raptor.ukf_utils import *
import random


if __name__ == '__main__':
    np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
    print("hello")

    gt_bag = "/mounted_folder/nocs/test/scene_1.bag"#; // Rosbag location & name for GT data
    est_bag = "/mounted_folder/raptor_processed_bags/nocs_test/msl_raptor_output_from_bag_scene_1.bag"#; // Rosbag location & name for EST data
 
    try:
        gt_bag = rosbag.Bag(gt_bag, 'r')
    except Exception as e:
        raise RuntimeError("Unable to Process GT Rosbag!!\n{}".format(e))
    try:
        est_bag = rosbag.Bag(est_bag, 'r')
    except Exception as e:
        raise RuntimeError("Unable to Process EST Rosbag!!\n{}".format(e))

    # pdb.set_trace()

    gt_topics = ["bowl_white_small_norm/mavros/local_position/pose", "bowl_white_small_norm/mavros/vision_pose/pose",
                 "camera_canon_len_norm/mavros/local_position/pose", "camera_canon_len_norm/mavros/vision_pose/pose",
                 "can_arizona_tea_norm/mavros/local_position/pose", "can_arizona_tea_norm/mavros/vision_pose/pose",
                 "laptop_air_xin_norm/mavros/local_position/pose", "laptop_air_xin_norm/mavros/vision_pose/pose",
                 "mug_daniel_norm/mavros/local_position/pose", "mug_daniel_norm/mavros/vision_pose/pose"]
    for topic, msg, t in gt_bag.read_messages(topics=gt_topics):
        print(msg)

    # est_topics = gt_topics
    # est_topics.extend()
    gt_bag.close()
    est_bag.close()

    print("done")