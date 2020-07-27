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
# Utils
# sys.path.append('/root/msl_raptor_ws/src/msl_raptor/src/utils_msl_raptor')
# from ssp_utils import *
# from math_utils import *
# from ros_utils import *
# from ukf_utils import *
# from raptor_logger import *
# from pose_metrics import *
# from viz_utils import *

class rosbag_combiner:
    def __init__(self, rb_path, rb_names, rb_namespaces, ego_ns, rb_out_name):
        bag_out = rosbag.Bag(rb_path + rb_out_name, 'w')
        b_first_bag = True
        gt_pose_msgs = []  # ns + mavros/vision_pose/pose
        est_pose_msgs = [] # ns + mavros/local_position/pose
        tracked_objs = TrackedObjects()
        all_bb_data = AngledBboxes()
        # common_topics = ["/" + ego_ns + "/camera/camera_info",
        #                  "/" + ego_ns + "/camera/image_raw",
        #                  "/" + ego_ns + "/mavros/vision_pose/pose",
        #                  "/" + ego_ns + "/mavros/local_position/pose"]
        for rb_name, rb_ns in zip(rb_names, rb_namespaces):
            topics = ["/" + rb_ns + "/mavros/local_position/pose", 
                      "/" + rb_ns + "/mavros/vision_pose/pose", 
                      "/" + ego_ns + "/msl_raptor_state",
                      "/" + ego_ns + "/bb_data",
                      "/" + ego_ns + "/camera/camera_info",
                      "/" + ego_ns + "/camera/image_raw",
                      "/" + ego_ns + "/mavros/vision_pose/pose",
                      "/" + ego_ns + "/mavros/local_position/pose"]
            
            bag_in = rosbag.Bag(rb_path + rb_name, 'r')
            # if b_first_bag:
            #     topics.extend(common_topics)
            for topic, msg, t in bag_in.read_messages(topics=topics):
                print(msg)
                if topic.split('/')[-1] == "msl_raptor_state":
                    # all_tracked_obj.append(msg.TrackedObject)
                    # pdb.set_trace()
                    # tracked_objs = TrackedObjects()
                    # tracked_objs.tracked_objects = msg.tracked_objects
                    bag_out.write("/" + ego_ns + "/msl_raptor_state", msg)
                    # all_tracked_obj.tracked_objects = [msg.TrackedObject]
                elif topic.split('/')[-1] == "bb_data":
                    # pdb.set_trace()
                    # angled_bb_boxes = AngledBboxes()
                    # angled_bb_boxes.boxes = msg.boxes
                    bag_out.write("/" + ego_ns + "/bb_data", msg)
                else:
                    bag_out.write(topic, msg)
            pdb.set_trace()
            bag_in.close()
            b_first_bag = False
        # bag_out.write("/" + ego_ns + "/msl_raptor_state", all_tracked_obj)
        bag_out.close()


if __name__ == '__main__':
    try:
        np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
        # program = rosbags_to_logs(rb_name=my_rb_name, data_source=my_data_source, ego_yaml=my_ego_yaml, ado_yaml=my_ado_yaml, b_save_3dbb_imgs=my_b_save_3dbb_imgs)
        rb_path = "/mounted_folder/raptor_processed_bags/nocs_test/"
        rb_names = ["msl_raptor_output_from_bag_scene_1_bowl_while_small_norm.bag",
                    "msl_raptor_output_from_bag_scene_1_camera_canon_len_norm.bag",
                    "msl_raptor_output_from_bag_scene_1_can_arizona_tea_norm.bag",
                    "msl_raptor_output_from_bag_scene_1_laptop_air_xin_norm.bag",
                    "msl_raptor_output_from_bag_scene_1_mug_daniel_norm.bag"]
        rb_namespaces = ["bowl_while_small_norm",
                        "camera_canon_len_norm",
                        "can_arizona_tea_norm",
                        "laptop_air_xin_norm",
                        "mug_daniel_norm"]
        ego_ns = "quad7"
        rb_out_name = "msl_raptor_output_from_bag_scene_1_merged.bag"

        rb_comb = rosbag_combiner(rb_path, rb_names, rb_namespaces, ego_ns, rb_out_name)
    except:
        import traceback
        traceback.print_exc()
    print("done with program!")
