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

class rosbag_vrpn_unifier:
    def __init__(self, rb_path_and_name, ego_ns, b_ego_pose_est):
        bag_out = rosbag.Bag(rb_path_and_name + "_unified.bag", 'w')
        bag_in = rosbag.Bag(rb_path_and_name + ".bag", 'r')
        for topic, msg, t in bag_in.read_messages():
            # print(t)
            if topic.split('/')[1] == "vrpn_client_node":
                ns_tmp = topic.split('/')[2]
                bag_out.write("/" + ns_tmp + "/mavros/vision_pose/pose", msg, t)
                if b_ego_pose_est and ns_tmp == ego_ns:
                    bag_out.write("/" + ns_tmp + "/mavros/local_position/pose", msg, t)
            else:
                bag_out.write(topic, msg, t)
        bag_in.close()
        b_first_bag = False
        bag_out.close()


        # bag_out = rosbag.Bag(rb_path + rb_out_name, 'r')
        # for topic, msg, t in bag_out.read_messages():
        #     print("time: {}, topic: {}".format(t, topic))
        bag_out.close()


if __name__ == '__main__':
    try:
        np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
        # program = rosbags_to_logs(rb_name=my_rb_name, data_source=my_data_source, ego_yaml=my_ego_yaml, ado_yaml=my_ado_yaml, b_save_3dbb_imgs=my_b_save_3dbb_imgs)
        rb_path_and_name_ = "/mounted_folder/rosbags_for_post_process/rosbag_for_post_process_2020-09-01-12-29-11"
        ego_ns_ = "quad7"
        b_ego_pose_est_ = True

        rb_unified = rosbag_vrpn_unifier(rb_path_and_name=rb_path_and_name_, ego_ns=ego_ns_, b_ego_pose_est=b_ego_pose_est_)
    except:
        import traceback
        traceback.print_exc()
    print("done with program!")